#include "prob.H"

namespace ProbParm {
AMREX_GPU_DEVICE_MANAGED amrex::Real p0 = 1.013e6; // [erg cm^-3]
AMREX_GPU_DEVICE_MANAGED amrex::Real T0 = 300.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real rho0 = 0.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real v0 = 0.;
AMREX_GPU_DEVICE_MANAGED amrex::Real inlet_vel = 2320.; // Unused
AMREX_GPU_DEVICE_MANAGED amrex::Real jet_vel = 5000.;
AMREX_GPU_DEVICE_MANAGED amrex::Real jet_dia = 1.E-2;
AMREX_GPU_DEVICE_MANAGED amrex::Real part_mean_dia = 1.E-3;
AMREX_GPU_DEVICE_MANAGED amrex::Real part_stdev_dia = 1.E-4;
AMREX_GPU_DEVICE_MANAGED amrex::Real mass_flow_rate = 1.;
AMREX_GPU_DEVICE_MANAGED amrex::Real part_rho = 0.640;
AMREX_GPU_DEVICE_MANAGED amrex::Real part_temp = 300.;
AMREX_GPU_DEVICE_MANAGED amrex::Real Y_O2 = 0.233;
AMREX_GPU_DEVICE_MANAGED amrex::Real Y_N2 = 0.767;
AMREX_GPU_DEVICE_MANAGED amrex::Real jet_start_time = 0.;
AMREX_GPU_DEVICE_MANAGED amrex::Real jet_end_time = 10000.;
AMREX_GPU_DEVICE_MANAGED amrex::Real spray_angle = 20.;
} // namespace ProbParm

extern "C" {
void
amrex_probinit(
  const int* init,
  const int* name,
  const int* namelen,
  const amrex_real* problo,
  const amrex_real* probhi)
{
  // Parse params
  amrex::ParmParse pp("prob");
  pp.query("init_v", ProbParm::v0);
  pp.query("ref_p", ProbParm::p0);
  pp.query("ref_T", ProbParm::T0);
  pp.query("init_N2", ProbParm::Y_N2);
  pp.query("init_O2", ProbParm::Y_O2);
  pp.query("inlet_vel", ProbParm::inlet_vel); // Currently unused
  pp.query("jet_vel", ProbParm::jet_vel);
  pp.query("jet_dia", ProbParm::jet_dia);
  pp.query("part_mean_dia", ProbParm::part_mean_dia);
  pp.query("part_stdev_dia", ProbParm::part_stdev_dia);
  pp.query("part_rho", ProbParm::part_rho);
  pp.query("part_temp", ProbParm::part_temp);
  pp.query("mass_flow_rate", ProbParm::mass_flow_rate);
  pp.query("spray_angle_deg", ProbParm::spray_angle);
  // Convert to radians
  ProbParm::spray_angle *= M_PI/180.;

  // Initial density, velocity, and material properties
  amrex::Real eint, cs, cp;
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  massfrac[N2_ID] = ProbParm::Y_N2;
  massfrac[O2_ID] = ProbParm::Y_O2;
  EOS::PYT2RE(ProbParm::p0, massfrac, ProbParm::T0, ProbParm::rho0, eint);
  EOS::RTY2Cs(ProbParm::rho0, ProbParm::T0, massfrac, cs);
  EOS::TY2Cp(ProbParm::T0, massfrac, cp);
}
}

void pc_prob_close()
{}
