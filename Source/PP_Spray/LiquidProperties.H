#ifndef LIQUIDPROPERTIES_H
#define LIQUIDPROPERTIES_H

using namespace amrex;

// Structs for liquid physical data
namespace pele::mp{

  namespace liqprop{

    /*
    struct LiqPropBase //: public physics::Factory<LiqPropBase>
    {
    public:

      static std::string base_identifier() { return "LiqPropBase"; }

      virtual amrex::Real getDensity() = 0;

      virtual ~LiqPropBase();

      amrex::Real dogs;
      amrex::Real cats[5];
    };
    */


    template <typename LPType>
    struct InitLiqProps
    {
    };

    struct ConstLiqProp // : LiqPropBase //::Register<LiqPropBase>
    {
    public:

      static std::string identifier() { return "ConstLiqProp"; }

      // Density of a specific liquid component
      // TODO: Add a function to compute mixture density
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real T2rho_i(const amrex::Real T, const int spf) const {
        amrex::Real a = rho_coef[4 * spf];
        amrex::Real b = rho_coef[4 * spf + 1];
        amrex::Real c = rho_coef[4 * spf + 2];
        amrex::Real d = rho_coef[4 * spf + 3];
        return a + T * (b + T * (c + T * d));
      }

      // Thermal conductivity of a specific liquid component
      // TODO: Add a function to compute mixture thermal conductivity
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real T2lambda_i(const amrex::Real& T, const int spf) const
      {
        amrex::Real a = lambda_coef[4 * spf];
        amrex::Real b = lambda_coef[4 * spf + 1];
        amrex::Real c = lambda_coef[4 * spf + 2];
        amrex::Real d = lambda_coef[4 * spf + 3];
        return a + T * (b + T * (c + T * d));
      }

      // Viscosity of a specific liquid component
      // TODO: Add a function to compute mixture viscosity
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real T2mu_i(const amrex::Real& T, const int spf) const
      {
        amrex::Real a = mu_coef[4 * spf];
        amrex::Real b = mu_coef[4 * spf + 1];
        amrex::Real c = mu_coef[4 * spf + 2];
        amrex::Real d = mu_coef[4 * spf + 3];
        return a + ((d / T + c) / T + b) / T;
      }

      // Whether or not to use Clausius Clapeyron (external) or Antoine (internal)
      // TODO: bring all psat stuff into a single function
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      bool use_psat_claus_clap(const int spf) const
      {
        return (psat_coef[4 * spf + 3] == 0.);
      }

      // Saturation vapor pressure of a specific liquid component (antoine eq.)
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real T2psat(const amrex::Real& T, const int spf) const
      {
        amrex::Real a = psat_coef[4 * spf];
        amrex::Real b = psat_coef[4 * spf + 1];
        amrex::Real c = psat_coef[4 * spf + 2];
        amrex::Real d = psat_coef[4 * spf + 3];
        return d * std::pow(10., a - b / (T + c));
      }

      /*
      // Estimate the boil temperature
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      void calcBoilT(const GasPhaseVals& gpv, amrex::Real* cBoilT) const
      {
        SprayUnits SPU;
        amrex::Real RU = pele::physics::Constants::RU * SPU.ru_conv;
        amrex::Real PATM = pele::physics::Constants::PATM * SPU.pres_conv;
        for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
          const int fspec = indx[spf];
          const amrex::Real mw_fuel = gpv.mw[fspec];
          // Since we only know the latent heat at the reference temperature,
          // modify Watsons power law to find latent heat at boiling conditions
          amrex::Real Hboil_ref =
            ref_latent[spf] *
            std::pow((critT[spf] - ref_T) / (critT[spf] - boilT[spf]), -0.38);
          // Estimate the boiling temperature at the gas phase pressure using
          // Clasius-Clapeyron relation
          cBoilT[spf] =
            1. / (std::log(PATM / gpv.p_fluid) * RU / (Hboil_ref * mw_fuel) +
                  1. / boilT[spf]);
          cBoilT[spf] = amrex::min(critT[spf], cBoilT[spf]);
        }
      }
      */

      // Component critical temperature
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real critT_i(const int spf) const
      {
        // TODO: Should we just take this from the gas phase computations?
        return critT[spf];
      }

      // Component heat capacity
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real T2Cp_i(const amrex::Real /*T*/, const int spf) const
      {
        return cp[spf];
      }

      // Component Tboil,Ref
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real Tboil_i(const int spf) const
      {
        return boilT[spf];
      }

      // Component latent heat of vaporization
      AMREX_GPU_HOST_DEVICE
      AMREX_FORCE_INLINE
      amrex::Real T2latent_i (const amrex::Real /*T*/, const int spf) const
      {
        return latent[spf];
      }

    private:
      // All contained data must be trivially copyable statically sized arrays
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM * 4> rho_coef;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM * 4> lambda_coef;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM * 4> mu_coef;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM * 4> psat_coef;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> critT;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> boilT;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> cp;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> latent;
      amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> ref_latent;

      friend struct InitLiqProps<ConstLiqProp>;

    };

    // This will be set somewhere else, potentially at runtime, rather than hardcoded
    using LiqPropType = ConstLiqProp;

    // Initializer functions
    template <>
    struct InitLiqProps<ConstLiqProp>
    {
      void operator()(LiqPropType* ldata, std::vector<std::string>& fuel_names)
      {
        amrex::ParmParse pp("spray");
        getInpCoef(ldata->rho_coef.data(), pp, fuel_names.data(), "rho", true);
        getInpCoef(ldata->mu_coef.data(), pp, fuel_names.data(), "mu");
        getInpCoef(ldata->lambda_coef.data(), pp, fuel_names.data(), "lambda");
        getInpCoef(ldata->psat_coef.data(), pp, fuel_names.data(), "psat");
        getInpVal(ldata->critT.data(), pp, fuel_names.data(), "crit_temp");
        getInpVal(ldata->boilT.data(), pp, fuel_names.data(), "boil_temp");
        getInpVal(ldata->cp.data(), pp, fuel_names.data(), "cp");
        getInpVal(ldata->ref_latent.data(), pp, fuel_names.data(), "latent");
        for (int i = 0; i < SPRAY_FUEL_NUM; ++i) {
          ldata->latent[i] = ldata->ref_latent[i];
        }
      }

      // TODO: Fix this nonsense
      void
getInpCoef(
  Real* coef,
  const ParmParse& ppp,
  const std::string* fuel_names,
  const std::string& varname,
  bool is_required = false)
{
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    std::string var_read = fuel_names[spf] + "_" + varname;
    int numvals = ppp.countval(var_read.c_str());
    // If 4 values are specified, assume fit coefficients
    if (numvals == 4) {
      std::vector<Real> inp_coef(4, 0.);
      if (is_required) {
        ppp.getarr(var_read.c_str(), inp_coef);
      } else {
        ppp.queryarr(var_read.c_str(), inp_coef);
      }
      for (int i = 0; i < 4; ++i) {
        coef[4 * spf + i] = inp_coef[i];
      }
    } else if (numvals == 1) {
      // If 1 value is specified, assume constant value
      Real inp_coef = 0.;
      for (int i = 0; i < 4; ++i) {
        coef[4 * spf + i] = 0.;
      }
      if (is_required) {
        ppp.get(var_read.c_str(), inp_coef);
      } else {
        ppp.query(var_read.c_str(), inp_coef);
      }
      coef[4 * spf] = inp_coef;
    }
  }
}

void
getInpVal(
  Real* coef,
  const ParmParse& ppp,
  const std::string* fuel_names,
  const std::string& varname)
{
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    std::string var_read = fuel_names[spf] + "_" + varname;
    ppp.get(var_read.c_str(), coef[spf]);
  }
}

    };

  }
}

#endif
