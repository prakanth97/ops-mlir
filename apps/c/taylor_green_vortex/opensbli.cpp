#include "constants.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define OPS_3D
#define OPS_API 2

#include "ops/OPSWrapper.h"

#include "io.h"
#include "opensbliblock00_kernels.h"
// #include "reductions.h" // uses ACC<double> + ops_arg_reduce; ops_arg_reduce is not modeled by ops_to_stencil.py yet

FILE *f0 = fopen("block0_output.log", "a");


int main(int argc, char **argv) {
  // Register kernel source file with JITEngine
  {
    std::string thisFile = __FILE__;
    std::string dir = thisFile.substr(0, thisFile.find_last_of('/'));
    set_kernel_source_file(dir + "/opensbliblock00_kernels.h");
  }
  
  // Initializing OPS
  ops_init(argc, argv, 2);
  // Set restart to 1 to restart the simulation from HDF5 file
  restart = 0;
  // User defined constant values
  block0np0 = 128;
  block0np1 = 128;
  block0np2 = 128;
  Delta0block0 = 2 * M_PI / block0np0;
  Delta1block0 = 2 * M_PI / block0np1;
  Delta2block0 = 2 * M_PI / block0np2;
  niter = 5000;
  double rkB[] = {(1.0 / 3.0), (15.0 / 16.0), (8.0 / 15.0)};
  double rkA[] = {0, (-5.0 / 9.0), (-153.0 / 128.0)};
  dt = 0.003385;
  write_output_file = 10000;
  HDF5_timing = 0;
  filter_frequency = 25;
  DRP_filt = 0.200000000000000;
  Minf = 0.1;
  gama = 1.4;
  Pr = 0.71;
  Re = 1600.0;
  inv2Delta0block0 = 1.0 / (Delta0block0 * Delta0block0);
  inv2Delta1block0 = 1.0 / (Delta1block0 * Delta1block0);
  inv2Delta2block0 = 1.0 / (Delta2block0 * Delta2block0);
  inv2Minf = 1.0 / (Minf * Minf);
  invDelta0block0 = 1.0 / (Delta0block0);
  invDelta1block0 = 1.0 / (Delta1block0);
  invDelta2block0 = 1.0 / (Delta2block0);
  invPr = 1.0 / (Pr);
  invRe = 1.0 / (Re);
  inv_gamma_m1 = 1.0 / ((-1 + gama));
  ops_decl_const("DRP_filt", 1, "double", &DRP_filt);
  ops_decl_const("Delta0block0", 1, "double", &Delta0block0);
  ops_decl_const("Delta1block0", 1, "double", &Delta1block0);
  ops_decl_const("Delta2block0", 1, "double", &Delta2block0);
  ops_decl_const("HDF5_timing", 1, "int", &HDF5_timing);
  ops_decl_const("Minf", 1, "double", &Minf);
  ops_decl_const("Pr", 1, "double", &Pr);
  ops_decl_const("Re", 1, "double", &Re);
  ops_decl_const("block0np0", 1, "int", &block0np0);
  ops_decl_const("block0np1", 1, "int", &block0np1);
  ops_decl_const("block0np2", 1, "int", &block0np2);
  ops_decl_const("dt", 1, "double", &dt);
  ops_decl_const("filter_frequency", 1, "int", &filter_frequency);
  ops_decl_const("gama", 1, "double", &gama);
  ops_decl_const("inv2Delta0block0", 1, "double", &inv2Delta0block0);
  ops_decl_const("inv2Delta1block0", 1, "double", &inv2Delta1block0);
  ops_decl_const("inv2Delta2block0", 1, "double", &inv2Delta2block0);
  ops_decl_const("inv2Minf", 1, "double", &inv2Minf);
  ops_decl_const("invDelta0block0", 1, "double", &invDelta0block0);
  ops_decl_const("invDelta1block0", 1, "double", &invDelta1block0);
  ops_decl_const("invDelta2block0", 1, "double", &invDelta2block0);
  ops_decl_const("invPr", 1, "double", &invPr);
  ops_decl_const("invRe", 1, "double", &invRe);
  ops_decl_const("inv_gamma_m1", 1, "double", &inv_gamma_m1);
  ops_decl_const("niter", 1, "int", &niter);
  ops_decl_const("simulation_time", 1, "double", &simulation_time);
  ops_decl_const("start_iter", 1, "int", &start_iter);
  ops_decl_const("write_output_file", 1, "int", &write_output_file);

  // Add const values to the kernel constant registry for translation into MLIR for the GPU backend
  ops_register_kernel_constant("DRP_filt", &DRP_filt);
  ops_register_kernel_constant("Delta0block0", &Delta0block0);
  ops_register_kernel_constant("Delta1block0", &Delta1block0);
  ops_register_kernel_constant("Delta2block0", &Delta2block0);
  ops_register_kernel_constant("HDF5_timing", &HDF5_timing);
  ops_register_kernel_constant("Minf", &Minf);
  ops_register_kernel_constant("Pr", &Pr);
  ops_register_kernel_constant("Re", &Re);
  ops_register_kernel_constant("block0np0", &block0np0);
  ops_register_kernel_constant("block0np1", &block0np1);
  ops_register_kernel_constant("block0np2", &block0np2);
  ops_register_kernel_constant("dt", &dt);
  ops_register_kernel_constant("filter_frequency", &filter_frequency);
  ops_register_kernel_constant("gama", &gama);
  ops_register_kernel_constant("inv2Delta0block0", &inv2Delta0block0);
  ops_register_kernel_constant("inv2Delta1block0", &inv2Delta1block0);
  ops_register_kernel_constant("inv2Delta2block0", &inv2Delta2block0);
  ops_register_kernel_constant("inv2Minf", &inv2Minf);
  ops_register_kernel_constant("invDelta0block0", &invDelta0block0);
  ops_register_kernel_constant("invDelta1block0", &invDelta1block0);
  ops_register_kernel_constant("invDelta2block0", &invDelta2block0);
  ops_register_kernel_constant("invPr", &invPr);
  ops_register_kernel_constant("invRe", &invRe);
  ops_register_kernel_constant("inv_gamma_m1", &inv_gamma_m1);
  ops_register_kernel_constant("niter", &niter);
  ops_register_kernel_constant("simulation_time", &simulation_time);
  ops_register_kernel_constant("start_iter", &start_iter);
  ops_register_kernel_constant("write_output_file", &write_output_file);

  // Define and Declare OPS Block
  ops_block opensbliblock00 = ops_decl_block(3, "opensbliblock00");
#include "defdec_data_set.h"
// Define and declare stencils
#include "bc_exchanges.h"
#include "stencils.h"
  // Init OPS partition
  double partition_start0, elapsed_partition_start0, partition_end0,
      elapsed_partition_end0;
  ops_timers(&partition_start0, &elapsed_partition_start0);
  ops_partition("");
  ops_timers(&partition_end0, &elapsed_partition_end0);
  ops_printf(
      "-----------------------------------------\n MPI partition and reading "
      "input file time: %lf\n -----------------------------------------\n",
      elapsed_partition_end0 - elapsed_partition_start0);
  // Restart procedure
  ops_printf("\033[1;32m");
  if (restart == 1) {
    ops_printf("OpenSBLI is restarting from the input file: restart.h5\n");
  } else {
    ops_printf("OpenSBLI is starting from the initial condition.\n");
  }
  ops_printf("\033[0m");
  // Constants from HDF5 restart file
  if (restart == 1) {
    ops_get_const_hdf5("simulation_time", 1, "double", (char *)&simulation_time,
                       "restart.h5");
    ops_get_const_hdf5("iter", 1, "int", (char *)&start_iter, "restart.h5");
  } else {
    simulation_time = 0.0;
    start_iter = 0;
  }
  tstart = simulation_time;

  if (restart == 0) {
    int iteration_range_39_block0[] = {-5, block0np0 + 5, -5, block0np1 + 5,
                                       -5, block0np2 + 5};
    ops_par_loop(
        opensbliblock00Kernel039, "opensbliblock00Kernel039", opensbliblock00,
        3, iteration_range_39_block0,
        ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
        ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
        ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
        ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
        ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE),
        ops_arg_idx());
    compile_and_execute();
  }

  // Initialize loop timers
  double cpu_start0, elapsed_start0, cpu_end0, elapsed_end0;
  ops_timers(&cpu_start0, &elapsed_start0);
  double inner_start, elapsed_inner_start;
  double inner_end, elapsed_inner_end;
  ops_timers(&inner_start, &elapsed_inner_start);
  for (iter = start_iter; iter <= start_iter + niter - 1; iter++) {
    simulation_time = tstart + dt * ((iter - start_iter) + 1);
    if (fmod(iter + 1, 100) == 0) {
      ops_timers(&inner_end, &elapsed_inner_end);
      ops_printf("Iteration: %d. Time-step: %.3e. Simulation time: %.5f. "
                 "Time/iteration: %lf.\n",
                 iter + 1, dt, simulation_time,
                 (elapsed_inner_end - elapsed_inner_start) / 100);
      ops_NaNcheck(rho_B0);
      ops_timers(&inner_start, &elapsed_inner_start);
    }

    ops_halo_transfer(periodicBC_direction0_side0_33_block0);
    ops_halo_transfer(periodicBC_direction0_side1_34_block0);
    ops_halo_transfer(periodicBC_direction1_side0_35_block0);
    ops_halo_transfer(periodicBC_direction1_side1_36_block0);
    ops_halo_transfer(periodicBC_direction2_side0_37_block0);
    ops_halo_transfer(periodicBC_direction2_side1_38_block0);

    for (stage = 0; stage <= 2; stage++) {
      int iteration_range_8_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2,
                                        -2, block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel008, "opensbliblock00Kernel008", opensbliblock00, 3,
          iteration_range_8_block0,
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_10_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2,
                                         -2, block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel010, "opensbliblock00Kernel010", opensbliblock00, 3,
          iteration_range_10_block0,
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_12_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2,
                                         -2, block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel012, "opensbliblock00Kernel012", opensbliblock00, 3,
          iteration_range_12_block0,
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_19_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2,
                                         -2, block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel019, "opensbliblock00Kernel019", opensbliblock00, 3,
          iteration_range_19_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_25_block0[] = {-2, block0np0 + 2, -2, block0np1 + 2,
                                         -2, block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel025, "opensbliblock00Kernel025", opensbliblock00, 3,
          iteration_range_25_block0,
          ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(T_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_7_block0[] = {0,  block0np0,    -2, block0np1 + 2,
                                        -2, block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel007, "opensbliblock00Kernel007",
          opensbliblock00, 3, iteration_range_7_block0,
          ops_arg_dat(u0_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
          ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_9_block0[] = {0, block0np0, -2, block0np1 + 2,
                                        0, block0np2};
      ops_par_loop(
          opensbliblock00Kernel009, "opensbliblock00Kernel009",
          opensbliblock00, 3, iteration_range_9_block0,
          ops_arg_dat(u1_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
          ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_11_block0[] = {0,         block0np0, 0,
                                         block0np1, -2,        block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel011, "opensbliblock00Kernel011",
          opensbliblock00, 3, iteration_range_11_block0,
          ops_arg_dat(u2_B0, 1, stencil_0_22_00_00_8, "double", OPS_READ),
          ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_13_block0[] = {0,         block0np0, 0,
                                         block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel013, "opensbliblock00Kernel013",
          opensbliblock00, 3, iteration_range_13_block0,
          ops_arg_dat(u0_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
          ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_14_block0[] = {0,         block0np0, 0,
                                         block0np1, -2,        block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel014, "opensbliblock00Kernel014",
          opensbliblock00, 3, iteration_range_14_block0,
          ops_arg_dat(u1_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
          ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_15_block0[] = {0,         block0np0, 0,
                                         block0np1, -2,        block0np2 + 2};
      ops_par_loop(
          opensbliblock00Kernel015, "opensbliblock00Kernel015",
          opensbliblock00, 3, iteration_range_15_block0,
          ops_arg_dat(u2_B0, 1, stencil_0_00_22_00_8, "double", OPS_READ),
          ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_16_block0[] = {0,         block0np0, 0,
                                         block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel016, "opensbliblock00Kernel016",
          opensbliblock00, 3, iteration_range_16_block0,
          ops_arg_dat(u0_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
          ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_17_block0[] = {0,         block0np0, 0,
                                         block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel017, "opensbliblock00Kernel017",
          opensbliblock00, 3, iteration_range_17_block0,
          ops_arg_dat(u1_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
          ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_18_block0[] = {0,         block0np0, 0,
                                         block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel018, "opensbliblock00Kernel018",
          opensbliblock00, 3, iteration_range_18_block0,
          ops_arg_dat(u2_B0, 1, stencil_0_00_00_22_8, "double", OPS_READ),
          ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_WRITE));

      int iteration_range_31_block0[] = {0,         block0np0, 0,
                                         block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel031, "opensbliblock00Kernel031", opensbliblock00, 3,
          iteration_range_31_block0,
          ops_arg_dat(p_B0, 1, stencil_0_22_22_22_24, "double", OPS_READ),
          ops_arg_dat(rhoE_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(rhou0_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(rhou1_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(rhou2_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_22_00_00_11, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_00_22_00_11, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
          ops_arg_dat(wk0_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk1_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk2_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk4_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk5_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(Residual0_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE));

      int iteration_range_32_block0[] = {0,         block0np0, 0,
                                         block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel032, "opensbliblock00Kernel032", opensbliblock00, 3,
          iteration_range_32_block0,
          ops_arg_dat(T_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(u0_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(u1_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(u2_B0, 1, stencil_0_22_22_22_27, "double", OPS_READ),
          ops_arg_dat(wk0_B0, 1, stencil_0_00_22_22_19, "double", OPS_READ),
          ops_arg_dat(wk1_B0, 1, stencil_0_00_22_00_11, "double", OPS_READ),
          ops_arg_dat(wk2_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
          ops_arg_dat(wk3_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk4_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
          ops_arg_dat(wk5_B0, 1, stencil_0_00_00_22_11, "double", OPS_READ),
          ops_arg_dat(wk6_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk7_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(wk8_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
          ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));

      int iteration_range_40_block0[] = {0, block0np0, 0, block0np1, 0, block0np2};
      ops_par_loop(
          opensbliblock00Kernel040, "opensbliblock00Kernel040",
          opensbliblock00, 3, iteration_range_40_block0,
          ops_arg_dat(Residual0_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(Residual1_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(Residual2_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(Residual3_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(Residual4_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_RW),
          ops_arg_gbl(&rkA[stage], 1, "double", OPS_READ),
          ops_arg_gbl(&rkB[stage], 1, "double", OPS_READ)
        );

      compile_and_execute();

      ops_halo_transfer(periodicBC_direction0_side0_33_block0);
      ops_halo_transfer(periodicBC_direction0_side1_34_block0);
      ops_halo_transfer(periodicBC_direction1_side0_35_block0);
      ops_halo_transfer(periodicBC_direction1_side1_36_block0);
      ops_halo_transfer(periodicBC_direction2_side0_37_block0);
      ops_halo_transfer(periodicBC_direction2_side1_38_block0);
    }

    if (fmod(1 + iter, filter_frequency) == 0) {
      int iteration_range_0_block0[] = {0,         block0np0, 0,
                                        block0np1, 0,         block0np2};
      ops_par_loop(opensbliblock00Kernel000,
                   "opensbliblock00Kernel000",
                   opensbliblock00, 3, iteration_range_0_block0,
                   ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                               OPS_WRITE),
                   ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                               OPS_WRITE),
                   ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3,
                               "double", OPS_WRITE),
                   ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3,
                               "double", OPS_WRITE),
                   ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3,
                               "double", OPS_WRITE));
      compile_and_execute();
    }

    if (fmod(1 + iter, filter_frequency) == 0) {
      int iteration_range_1_block0[] = {0,         block0np0, 0,
                                        block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel001,
          "opensbliblock00Kernel001",
          opensbliblock00, 3, iteration_range_1_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_55_00_00_23, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_55_00_00_23, "double", OPS_READ),
          ops_arg_dat(rhou0_B0, 1, stencil_0_55_00_00_23, "double", OPS_READ),
          ops_arg_dat(rhou1_B0, 1, stencil_0_55_00_00_23, "double", OPS_READ),
          ops_arg_dat(rhou2_B0, 1, stencil_0_55_00_00_23, "double", OPS_READ),
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE));
      compile_and_execute();
    }

    if (fmod(1 + iter, filter_frequency) == 0) {
      int iteration_range_2_block0[] = {0,         block0np0, 0,
                                        block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel002,
          "opensbliblock00Kernel002",
          opensbliblock00, 3, iteration_range_2_block0,
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));
      compile_and_execute();
    }

    if (fmod(1 + iter, filter_frequency) == 0) {
      int iteration_range_3_block0[] = {0,         block0np0, 0,
                                        block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel003,
          "opensbliblock00Kernel003",
          opensbliblock00, 3, iteration_range_3_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_00_55_00_23, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_00_55_00_23, "double", OPS_READ),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00_55_00_23, "double", OPS_READ),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00_55_00_23, "double", OPS_READ),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00_55_00_23, "double", OPS_READ),
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE));
      compile_and_execute();
    }

    if (fmod(1 + iter, filter_frequency) == 0) {
      int iteration_range_4_block0[] = {0,         block0np0, 0,
                                        block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel004,
          "opensbliblock00Kernel004",
          opensbliblock00, 3, iteration_range_4_block0,
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));
      compile_and_execute();
    }

    if (fmod(1 + iter, filter_frequency) == 0) {
      int iteration_range_5_block0[] = {0,         block0np0, 0,
                                        block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel005,
          "opensbliblock00Kernel005",
          opensbliblock00, 3, iteration_range_5_block0,
          ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_55_23, "double", OPS_READ),
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_55_23, "double", OPS_READ),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_55_23, "double", OPS_READ),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_55_23, "double", OPS_READ),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_55_23, "double", OPS_READ),
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_WRITE));
      compile_and_execute();
    }

    if (fmod(1 + iter, filter_frequency) == 0) {
      int iteration_range_6_block0[] = {0,         block0np0, 0,
                                        block0np1, 0,         block0np2};
      ops_par_loop(
          opensbliblock00Kernel006,
          "opensbliblock00Kernel006",
          opensbliblock00, 3, iteration_range_6_block0,
          ops_arg_dat(rhoE_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rho_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou0_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou1_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhou2_RKold_B0, 1, stencil_0_00_00_00_3, "double",
                      OPS_READ),
          ops_arg_dat(rhoE_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rho_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou0_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou1_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW),
          ops_arg_dat(rhou2_B0, 1, stencil_0_00_00_00_3, "double", OPS_RW));
      compile_and_execute();
    }

    // NOT PORTED: this monitoring block needs ops_arg_reduce (OPS_INC),
    // which ops_to_stencil.py does not model yet -- see the comment at the
    // top of opensbliblock00_kernels.h and reductions.h.
    //
    // if (fmod(1 + iter, 100) == 0 || iter == 0) {
    //   // Data access for simulation monitoring
    //   if (iter == 0) {
    //     ops_fprintf(f0, "Iteration, Time, p_B0(30, 30, 30)\n");
    //   }
    //   // Monitoring of p_B0
    //   ops_reduction reduce_0_p_B0 = ops_decl_reduction_handle(
    //       sizeof(double), "double", "reduction_0_p_B0");
    //   double p_B0_0_output = 0.0;
    //   int i00 = 30, j00 = 30, k00 = 30;
    //   int monitor_range_0_p_B0[] = {i00, i00 + 1, j00, j00 + 1, k00, k00 + 1};
    //   ops_par_loop(
    //       monitor_0_p_B0, "monitor_0_p_B0", opensbliblock00, 3,
    //       monitor_range_0_p_B0,
    //       ops_arg_dat(p_B0, 1, stencil_0_00_00_00_3, "double", OPS_READ),
    //       ops_arg_reduce(reduce_0_p_B0, 1, "double", OPS_INC));
    //   compile_and_execute();
    //   ops_reduction_result(reduce_0_p_B0, &p_B0_0_output);
    //
    //   // Write the output values
    //   ops_fprintf(f0, "%d, %.12e, %.12e\n", iter + 1, simulation_time,
    //               p_B0_0_output);
    // }

    if (fmod(1 + iter, write_output_file) == 0 || iter == 0) {
      HDF5_IO_Write_0_opensbliblock00_dynamic(opensbliblock00, iter, rho_B0,
                                              rhou0_B0, rhou1_B0, rhou2_B0,
                                              rhoE_B0, HDF5_timing);
    }
  }
  ops_timers(&cpu_end0, &elapsed_end0);
  ops_printf("\nTimings are:\n");
  ops_printf("-----------------------------------------\n");
  ops_printf("Total Wall time %lf\n", elapsed_end0 - elapsed_start0);

  HDF5_IO_Write_0_opensbliblock00(opensbliblock00, rho_B0, rhou0_B0, rhou1_B0,
                                  rhou2_B0, rhoE_B0, HDF5_timing);
  ops_timing_output(std::cout);
  fclose(f0);
  ops_exit();
  // Main program end
}
