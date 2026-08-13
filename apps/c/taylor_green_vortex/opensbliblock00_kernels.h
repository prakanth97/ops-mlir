#ifndef OPENSBLIBLOCK00_KERNEL_H
#define OPENSBLIBLOCK00_KERNEL_H

#include "math.h"

// Declaration of global constants
extern int restart;
extern int iter;
extern int stage;
extern double tstart;
extern double DRP_filt;
extern double Delta0block0;
extern double Delta1block0;
extern double Delta2block0;
extern int HDF5_timing;
extern double Minf;
extern double Pr;
extern double Re;
extern int block0np0;
extern int block0np1;
extern int block0np2;
extern double dt;
extern int filter_frequency;
extern double gama;
extern double inv2Delta0block0;
extern double inv2Delta1block0;
extern double inv2Delta2block0;
extern double inv2Minf;
extern double invDelta0block0;
extern double invDelta1block0;
extern double invDelta2block0;
extern double invPr;
extern double invRe;
extern double inv_gamma_m1;
extern int niter;
extern double simulation_time;
extern int start_iter;
extern int write_output_file;

struct opensbliblock00Kernel039_result {
  double rhoE_B0;
  double rho_B0;
  double rhou0_B0;
  double rhou1_B0;
  double rhou2_B0;
};

void opensbliblock00Kernel039(const int *idx, opensbliblock00Kernel039_result *out)
{
   double x0 = Delta0block0*idx[0];
   double x1 = Delta1block0*idx[1];
   double x2 = Delta2block0*idx[2];

   double u0 = cos(x1)*cos(x2)*sin(x0);
   double u1 = -cos(x0)*cos(x2)*sin(x1);
   double u2 = 0.0;

   double p = (2.0 + cos(2.0*x2))*(0.0625*cos(2.0*x0) + 0.0625*cos(2.0*x1)) + 1.0/((Minf*Minf)*gama);
   double r = gama*p*(Minf*Minf);

   out->rhoE_B0 = p/(-1 + gama) + 0.5*r*((u0*u0) + (u1*u1) + (u2*u2));
   out->rho_B0 = r;
   out->rhou0_B0 = r*u0;
   out->rhou1_B0 = r*u1;
   out->rhou2_B0 = r*u2;
}

double opensbliblock00Kernel008(double rho_B0, double rhou0_B0)
{
   return rhou0_B0/rho_B0;
}

double opensbliblock00Kernel010(double rho_B0, double rhou1_B0)
{
   return rhou1_B0/rho_B0;
}

double opensbliblock00Kernel012(double rho_B0, double rhou2_B0)
{
   return rhou2_B0/rho_B0;
}

double opensbliblock00Kernel019(double rhoE_B0, double rho_B0, double u0_B0, double u1_B0, double u2_B0)
{
   return (-1 + gama)*(-(1.0/2.0)*(u0_B0*u0_B0)*rho_B0 -
     (1.0/2.0)*(u1_B0*u1_B0)*rho_B0 - (1.0/2.0)*(u2_B0*u2_B0)*rho_B0 +
     rhoE_B0);
}

double opensbliblock00Kernel025(double p_B0, double rho_B0)
{
   return (Minf*Minf)*gama*p_B0/rho_B0;
}

// stencil_0_22_00_00_8: {-2,0,0, -1,0,0, 1,0,0, 2,0,0}
double opensbliblock00Kernel007(double u0_B0_xm2, double u0_B0_xm1, double u0_B0_xp1, double u0_B0_xp2)
{
    return (-(2.0/3.0)*u0_B0_xm1 - (1.0/12.0)*u0_B0_xp2 + ((1.0/12.0))*u0_B0_xm2 +
      ((2.0/3.0))*u0_B0_xp1)*invDelta0block0;
}

double opensbliblock00Kernel009(double u1_B0_xm2, double u1_B0_xm1, double u1_B0_xp1, double u1_B0_xp2)
{
    return (-(2.0/3.0)*u1_B0_xm1 - (1.0/12.0)*u1_B0_xp2 + ((1.0/12.0))*u1_B0_xm2 +
      ((2.0/3.0))*u1_B0_xp1)*invDelta0block0;
}

double opensbliblock00Kernel011(double u2_B0_xm2, double u2_B0_xm1, double u2_B0_xp1, double u2_B0_xp2)
{
    return (-(2.0/3.0)*u2_B0_xm1 - (1.0/12.0)*u2_B0_xp2 + ((1.0/12.0))*u2_B0_xm2 +
      ((2.0/3.0))*u2_B0_xp1)*invDelta0block0;
}

// stencil_0_00_22_00_8: {0,-2,0, 0,-1,0, 0,1,0, 0,2,0}
double opensbliblock00Kernel013(double u0_B0_ym2, double u0_B0_ym1, double u0_B0_yp1, double u0_B0_yp2)
{
    return (-(2.0/3.0)*u0_B0_ym1 - (1.0/12.0)*u0_B0_yp2 + ((1.0/12.0))*u0_B0_ym2 +
      ((2.0/3.0))*u0_B0_yp1)*invDelta1block0;
}

double opensbliblock00Kernel014(double u1_B0_ym2, double u1_B0_ym1, double u1_B0_yp1, double u1_B0_yp2)
{
    return (-(2.0/3.0)*u1_B0_ym1 - (1.0/12.0)*u1_B0_yp2 + ((1.0/12.0))*u1_B0_ym2 +
      ((2.0/3.0))*u1_B0_yp1)*invDelta1block0;
}

double opensbliblock00Kernel015(double u2_B0_ym2, double u2_B0_ym1, double u2_B0_yp1, double u2_B0_yp2)
{
    return (-(2.0/3.0)*u2_B0_ym1 - (1.0/12.0)*u2_B0_yp2 + ((1.0/12.0))*u2_B0_ym2 +
      ((2.0/3.0))*u2_B0_yp1)*invDelta1block0;
}

// stencil_0_00_00_22_8: {0,0,-2, 0,0,-1, 0,0,1, 0,0,2}
double opensbliblock00Kernel016(double u0_B0_zm2, double u0_B0_zm1, double u0_B0_zp1, double u0_B0_zp2)
{
    return (-(2.0/3.0)*u0_B0_zm1 - (1.0/12.0)*u0_B0_zp2 + ((1.0/12.0))*u0_B0_zm2 +
      ((2.0/3.0))*u0_B0_zp1)*invDelta2block0;
}

double opensbliblock00Kernel017(double u1_B0_zm2, double u1_B0_zm1, double u1_B0_zp1, double u1_B0_zp2)
{
    return (-(2.0/3.0)*u1_B0_zm1 - (1.0/12.0)*u1_B0_zp2 + ((1.0/12.0))*u1_B0_zm2 +
      ((2.0/3.0))*u1_B0_zp1)*invDelta2block0;
}

double opensbliblock00Kernel018(double u2_B0_zm2, double u2_B0_zm1, double u2_B0_zp1, double u2_B0_zp2)
{
    return (-(2.0/3.0)*u2_B0_zm1 - (1.0/12.0)*u2_B0_zp2 + ((1.0/12.0))*u2_B0_zm2 +
      ((2.0/3.0))*u2_B0_zp1)*invDelta2block0;
}

// Convective terms. Read-arg stencils (stencils.h):
//   p_B0:               stencil_0_22_22_22_24 (12 pts, no center)
//   rhoE/rho/rhou0/1/2:  stencil_0_22_22_22_27 (13 pts, incl. center)
//   u0_B0:               stencil_0_22_00_00_11 (5 pts x, incl. center)
//   u1_B0:               stencil_0_00_22_00_11 (5 pts y, incl. center)
//   u2_B0:               stencil_0_00_00_22_11 (5 pts z, incl. center)
//   wk0_B0..wk8_B0:      stencil_0_00_00_00_3 (center only)
struct opensbliblock00Kernel031_result {
  double Residual0_B0;
  double Residual1_B0;
  double Residual2_B0;
  double Residual3_B0;
  double Residual4_B0;
};

void opensbliblock00Kernel031(
    double p_B0_xm2, double p_B0_xm1, double p_B0_ym2, double p_B0_ym1,
    double p_B0_zm2, double p_B0_zm1, double p_B0_zp1, double p_B0_zp2,
    double p_B0_yp1, double p_B0_yp2, double p_B0_xp1, double p_B0_xp2,
    double rhoE_B0_xm2, double rhoE_B0_xm1, double rhoE_B0_ym2, double rhoE_B0_ym1,
    double rhoE_B0_zm2, double rhoE_B0_zm1, double rhoE_B0_c, double rhoE_B0_zp1,
    double rhoE_B0_zp2, double rhoE_B0_yp1, double rhoE_B0_yp2, double rhoE_B0_xp1, double rhoE_B0_xp2,
    double rho_B0_xm2, double rho_B0_xm1, double rho_B0_ym2, double rho_B0_ym1,
    double rho_B0_zm2, double rho_B0_zm1, double rho_B0_c, double rho_B0_zp1,
    double rho_B0_zp2, double rho_B0_yp1, double rho_B0_yp2, double rho_B0_xp1, double rho_B0_xp2,
    double rhou0_B0_xm2, double rhou0_B0_xm1, double rhou0_B0_ym2, double rhou0_B0_ym1,
    double rhou0_B0_zm2, double rhou0_B0_zm1, double rhou0_B0_c, double rhou0_B0_zp1,
    double rhou0_B0_zp2, double rhou0_B0_yp1, double rhou0_B0_yp2, double rhou0_B0_xp1, double rhou0_B0_xp2,
    double rhou1_B0_xm2, double rhou1_B0_xm1, double rhou1_B0_ym2, double rhou1_B0_ym1,
    double rhou1_B0_zm2, double rhou1_B0_zm1, double rhou1_B0_c, double rhou1_B0_zp1,
    double rhou1_B0_zp2, double rhou1_B0_yp1, double rhou1_B0_yp2, double rhou1_B0_xp1, double rhou1_B0_xp2,
    double rhou2_B0_xm2, double rhou2_B0_xm1, double rhou2_B0_ym2, double rhou2_B0_ym1,
    double rhou2_B0_zm2, double rhou2_B0_zm1, double rhou2_B0_c, double rhou2_B0_zp1,
    double rhou2_B0_zp2, double rhou2_B0_yp1, double rhou2_B0_yp2, double rhou2_B0_xp1, double rhou2_B0_xp2,
    double u0_B0_xm2, double u0_B0_xm1, double u0_B0_c, double u0_B0_xp1, double u0_B0_xp2,
    double u1_B0_ym2, double u1_B0_ym1, double u1_B0_c, double u1_B0_yp1, double u1_B0_yp2,
    double u2_B0_zm2, double u2_B0_zm1, double u2_B0_c, double u2_B0_zp1, double u2_B0_zp2,
    double wk0_B0, double wk1_B0, double wk2_B0, double wk3_B0, double wk4_B0,
    double wk5_B0, double wk6_B0, double wk7_B0, double wk8_B0,
    opensbliblock00Kernel031_result *out)
{
   double d1_inv_rhoErho_dx = (-(2.0/3.0)*rhoE_B0_xm1/rho_B0_xm1 - (1.0/12.0)*rhoE_B0_xp2/rho_B0_xp2 +
     ((1.0/12.0))*rhoE_B0_xm2/rho_B0_xm2 + ((2.0/3.0))*rhoE_B0_xp1/rho_B0_xp1)*invDelta0block0;
   double d1_inv_rhoErho_dy = (-(2.0/3.0)*rhoE_B0_ym1/rho_B0_ym1 - (1.0/12.0)*rhoE_B0_yp2/rho_B0_yp2 +
     ((1.0/12.0))*rhoE_B0_ym2/rho_B0_ym2 + ((2.0/3.0))*rhoE_B0_yp1/rho_B0_yp1)*invDelta1block0;
   double d1_inv_rhoErho_dz = (-(2.0/3.0)*rhoE_B0_zm1/rho_B0_zm1 - (1.0/12.0)*rhoE_B0_zp2/rho_B0_zp2 +
     ((1.0/12.0))*rhoE_B0_zm2/rho_B0_zm2 + ((2.0/3.0))*rhoE_B0_zp1/rho_B0_zp1)*invDelta2block0;

   double d1_p_dx = (-(2.0/3.0)*p_B0_xm1 - (1.0/12.0)*p_B0_xp2 + ((1.0/12.0))*p_B0_xm2 +
     ((2.0/3.0))*p_B0_xp1)*invDelta0block0;
   double d1_p_dy = (-(2.0/3.0)*p_B0_ym1 - (1.0/12.0)*p_B0_yp2 + ((1.0/12.0))*p_B0_ym2 +
     ((2.0/3.0))*p_B0_yp1)*invDelta1block0;
   double d1_p_dz = (-(2.0/3.0)*p_B0_zm1 - (1.0/12.0)*p_B0_zp2 + ((1.0/12.0))*p_B0_zm2 +
     ((2.0/3.0))*p_B0_zp1)*invDelta2block0;

   double d1_pu0_dx = (-(2.0/3.0)*p_B0_xm1*u0_B0_xm1 - (1.0/12.0)*p_B0_xp2*u0_B0_xp2 +
     ((1.0/12.0))*p_B0_xm2*u0_B0_xm2 + ((2.0/3.0))*p_B0_xp1*u0_B0_xp1)*invDelta0block0;
   double d1_pu1_dy = (-(2.0/3.0)*p_B0_ym1*u1_B0_ym1 - (1.0/12.0)*p_B0_yp2*u1_B0_yp2 +
     ((1.0/12.0))*p_B0_ym2*u1_B0_ym2 + ((2.0/3.0))*p_B0_yp1*u1_B0_yp1)*invDelta1block0;
   double d1_pu2_dz = (-(2.0/3.0)*p_B0_zm1*u2_B0_zm1 - (1.0/12.0)*p_B0_zp2*u2_B0_zp2 +
     ((1.0/12.0))*p_B0_zm2*u2_B0_zm2 + ((2.0/3.0))*p_B0_zp1*u2_B0_zp1)*invDelta2block0;

   double d1_rhoEu0_dx = (-(2.0/3.0)*u0_B0_xm1*rhoE_B0_xm1 - (1.0/12.0)*u0_B0_xp2*rhoE_B0_xp2 +
     ((1.0/12.0))*u0_B0_xm2*rhoE_B0_xm2 + ((2.0/3.0))*u0_B0_xp1*rhoE_B0_xp1)*invDelta0block0;
   double d1_rhoEu1_dy = (-(2.0/3.0)*u1_B0_ym1*rhoE_B0_ym1 - (1.0/12.0)*u1_B0_yp2*rhoE_B0_yp2 +
     ((1.0/12.0))*u1_B0_ym2*rhoE_B0_ym2 + ((2.0/3.0))*u1_B0_yp1*rhoE_B0_yp1)*invDelta1block0;
   double d1_rhoEu2_dz = (-(2.0/3.0)*u2_B0_zm1*rhoE_B0_zm1 - (1.0/12.0)*u2_B0_zp2*rhoE_B0_zp2 +
     ((1.0/12.0))*u2_B0_zm2*rhoE_B0_zm2 + ((2.0/3.0))*u2_B0_zp1*rhoE_B0_zp1)*invDelta2block0;

   double d1_rhou0_dx = (-(2.0/3.0)*rhou0_B0_xm1 - (1.0/12.0)*rhou0_B0_xp2 + ((1.0/12.0))*rhou0_B0_xm2 +
     ((2.0/3.0))*rhou0_B0_xp1)*invDelta0block0;
   double d1_rhou1_dy = (-(2.0/3.0)*rhou1_B0_ym1 - (1.0/12.0)*rhou1_B0_yp2 + ((1.0/12.0))*rhou1_B0_ym2 +
     ((2.0/3.0))*rhou1_B0_yp1)*invDelta1block0;
   double d1_rhou2_dz = (-(2.0/3.0)*rhou2_B0_zm1 - (1.0/12.0)*rhou2_B0_zp2 + ((1.0/12.0))*rhou2_B0_zm2 +
     ((2.0/3.0))*rhou2_B0_zp1)*invDelta2block0;

   double d1_rhou0u0_dx = (-(2.0/3.0)*u0_B0_xm1*rhou0_B0_xm1 - (1.0/12.0)*u0_B0_xp2*rhou0_B0_xp2 +
     ((1.0/12.0))*u0_B0_xm2*rhou0_B0_xm2 + ((2.0/3.0))*u0_B0_xp1*rhou0_B0_xp1)*invDelta0block0;
   double d1_rhou1u0_dx = (-(2.0/3.0)*u0_B0_xm1*rhou1_B0_xm1 - (1.0/12.0)*u0_B0_xp2*rhou1_B0_xp2 +
     ((1.0/12.0))*u0_B0_xm2*rhou1_B0_xm2 + ((2.0/3.0))*u0_B0_xp1*rhou1_B0_xp1)*invDelta0block0;
   double d1_rhou2u0_dx = (-(2.0/3.0)*u0_B0_xm1*rhou2_B0_xm1 - (1.0/12.0)*u0_B0_xp2*rhou2_B0_xp2 +
     ((1.0/12.0))*u0_B0_xm2*rhou2_B0_xm2 + ((2.0/3.0))*u0_B0_xp1*rhou2_B0_xp1)*invDelta0block0;

   double d1_rhou0u1_dy = (-(2.0/3.0)*u1_B0_ym1*rhou0_B0_ym1 - (1.0/12.0)*u1_B0_yp2*rhou0_B0_yp2 +
     ((1.0/12.0))*u1_B0_ym2*rhou0_B0_ym2 + ((2.0/3.0))*u1_B0_yp1*rhou0_B0_yp1)*invDelta1block0;
   double d1_rhou1u1_dy = (-(2.0/3.0)*u1_B0_ym1*rhou1_B0_ym1 - (1.0/12.0)*u1_B0_yp2*rhou1_B0_yp2 +
     ((1.0/12.0))*u1_B0_ym2*rhou1_B0_ym2 + ((2.0/3.0))*u1_B0_yp1*rhou1_B0_yp1)*invDelta1block0;
   double d1_rhou2u1_dy = (-(2.0/3.0)*u1_B0_ym1*rhou2_B0_ym1 - (1.0/12.0)*u1_B0_yp2*rhou2_B0_yp2 +
     ((1.0/12.0))*u1_B0_ym2*rhou2_B0_ym2 + ((2.0/3.0))*u1_B0_yp1*rhou2_B0_yp1)*invDelta1block0;

   double d1_rhou0u2_dz = (-(2.0/3.0)*u2_B0_zm1*rhou0_B0_zm1 - (1.0/12.0)*u2_B0_zp2*rhou0_B0_zp2 +
     ((1.0/12.0))*u2_B0_zm2*rhou0_B0_zm2 + ((2.0/3.0))*u2_B0_zp1*rhou0_B0_zp1)*invDelta2block0;
   double d1_rhou1u2_dz = (-(2.0/3.0)*u2_B0_zm1*rhou1_B0_zm1 - (1.0/12.0)*u2_B0_zp2*rhou1_B0_zp2 +
     ((1.0/12.0))*u2_B0_zm2*rhou1_B0_zm2 + ((2.0/3.0))*u2_B0_zp1*rhou1_B0_zp1)*invDelta2block0;
   double d1_rhou2u2_dz = (-(2.0/3.0)*u2_B0_zm1*rhou2_B0_zm1 - (1.0/12.0)*u2_B0_zp2*rhou2_B0_zp2 +
     ((1.0/12.0))*u2_B0_zm2*rhou2_B0_zm2 + ((2.0/3.0))*u2_B0_zp1*rhou2_B0_zp1)*invDelta2block0;

   out->Residual0_B0 = -d1_rhou0_dx - d1_rhou1_dy - d1_rhou2_dz;

   out->Residual1_B0 = -d1_p_dx - (1.0/2.0)*d1_rhou0u0_dx - (1.0/2.0)*d1_rhou0u1_dy - (1.0/2.0)*d1_rhou0u2_dz -
     (1.0/2.0)*u0_B0_c*d1_rhou0_dx - (1.0/2.0)*u0_B0_c*d1_rhou1_dy - (1.0/2.0)*u0_B0_c*d1_rhou2_dz -
     (1.0/2.0)*wk0_B0*rhou0_B0_c - (1.0/2.0)*wk3_B0*rhou1_B0_c -
     (1.0/2.0)*wk6_B0*rhou2_B0_c;

   out->Residual2_B0 = -d1_p_dy - (1.0/2.0)*d1_rhou1u0_dx - (1.0/2.0)*d1_rhou1u1_dy - (1.0/2.0)*d1_rhou1u2_dz -
     (1.0/2.0)*(d1_rhou0_dx + d1_rhou1_dy + d1_rhou2_dz)*u1_B0_c - (1.0/2.0)*wk1_B0*rhou0_B0_c -
     (1.0/2.0)*wk4_B0*rhou1_B0_c - (1.0/2.0)*wk7_B0*rhou2_B0_c;

   out->Residual3_B0 = -d1_p_dz - (1.0/2.0)*d1_rhou2u0_dx - (1.0/2.0)*d1_rhou2u1_dy - (1.0/2.0)*d1_rhou2u2_dz -
     (1.0/2.0)*(d1_rhou0_dx + d1_rhou1_dy + d1_rhou2_dz)*u2_B0_c - (1.0/2.0)*wk2_B0*rhou0_B0_c -
     (1.0/2.0)*wk5_B0*rhou1_B0_c - (1.0/2.0)*wk8_B0*rhou2_B0_c;

   out->Residual4_B0 = -d1_pu0_dx - d1_pu1_dy - d1_pu2_dz - (1.0/2.0)*d1_rhoEu0_dx - (1.0/2.0)*d1_rhoEu1_dy -
     (1.0/2.0)*d1_rhoEu2_dz - (1.0/2.0)*rhou0_B0_c*d1_inv_rhoErho_dx - (1.0/2.0)*rhou1_B0_c*d1_inv_rhoErho_dy
     - (1.0/2.0)*rhou2_B0_c*d1_inv_rhoErho_dz - (1.0/2.0)*(d1_rhou0_dx + d1_rhou1_dy +
     d1_rhou2_dz)*rhoE_B0_c/rho_B0_c;
}

// Viscous terms. Read-arg stencils:
//   T_B0/u0_B0/u1_B0/u2_B0: stencil_0_22_22_22_27 (13 pts, incl. center)
//   wk0_B0:                 stencil_0_00_22_22_19 (9 pts, incl. center)
//   wk1_B0:                 stencil_0_00_22_00_11 (5 pts y, incl. center)
//   wk2_B0, wk4_B0, wk5_B0:  stencil_0_00_00_22_11 (5 pts z, incl. center)
//   wk3_B0, wk6_B0..wk8_B0: stencil_0_00_00_00_3 (center only)
//   Residual1_B0..Residual4_B0: OPS_RW on stencil_0_00_00_00_3 -- read AND written
struct opensbliblock00Kernel032_result {
  double Residual1_B0;
  double Residual2_B0;
  double Residual3_B0;
  double Residual4_B0;
};

void opensbliblock00Kernel032(
    double T_B0_xm2, double T_B0_xm1, double T_B0_ym2, double T_B0_ym1,
    double T_B0_zm2, double T_B0_zm1, double T_B0_c, double T_B0_zp1,
    double T_B0_zp2, double T_B0_yp1, double T_B0_yp2, double T_B0_xp1, double T_B0_xp2,
    double u0_B0_xm2, double u0_B0_xm1, double u0_B0_ym2, double u0_B0_ym1,
    double u0_B0_zm2, double u0_B0_zm1, double u0_B0_c, double u0_B0_zp1,
    double u0_B0_zp2, double u0_B0_yp1, double u0_B0_yp2, double u0_B0_xp1, double u0_B0_xp2,
    double u1_B0_xm2, double u1_B0_xm1, double u1_B0_ym2, double u1_B0_ym1,
    double u1_B0_zm2, double u1_B0_zm1, double u1_B0_c, double u1_B0_zp1,
    double u1_B0_zp2, double u1_B0_yp1, double u1_B0_yp2, double u1_B0_xp1, double u1_B0_xp2,
    double u2_B0_xm2, double u2_B0_xm1, double u2_B0_ym2, double u2_B0_ym1,
    double u2_B0_zm2, double u2_B0_zm1, double u2_B0_c, double u2_B0_zp1,
    double u2_B0_zp2, double u2_B0_yp1, double u2_B0_yp2, double u2_B0_xp1, double u2_B0_xp2,
    double wk0_B0_ym2, double wk0_B0_ym1, double wk0_B0_zm2, double wk0_B0_zm1,
    double wk0_B0_c, double wk0_B0_zp1, double wk0_B0_zp2, double wk0_B0_yp1, double wk0_B0_yp2,
    double wk1_B0_ym2, double wk1_B0_ym1, double wk1_B0_c, double wk1_B0_yp1, double wk1_B0_yp2,
    double wk2_B0_zm2, double wk2_B0_zm1, double wk2_B0_c, double wk2_B0_zp1, double wk2_B0_zp2,
    double wk3_B0,
    double wk4_B0_zm2, double wk4_B0_zm1, double wk4_B0_c, double wk4_B0_zp1, double wk4_B0_zp2,
    double wk5_B0_zm2, double wk5_B0_zm1, double wk5_B0_c, double wk5_B0_zp1, double wk5_B0_zp2,
    double wk6_B0, double wk7_B0, double wk8_B0,
    double Residual1_B0, double Residual2_B0, double Residual3_B0, double Residual4_B0,
    opensbliblock00Kernel032_result *out)
{
   double d2_T_dx = (-(5.0/2.0)*T_B0_c - (1.0/12.0)*T_B0_xm2 - (1.0/12.0)*T_B0_xp2 + ((4.0/3.0))*T_B0_xp1 +
     ((4.0/3.0))*T_B0_xm1)*inv2Delta0block0;
   double d2_u0_dx = (-(5.0/2.0)*u0_B0_c - (1.0/12.0)*u0_B0_xm2 - (1.0/12.0)*u0_B0_xp2 + ((4.0/3.0))*u0_B0_xp1
     + ((4.0/3.0))*u0_B0_xm1)*inv2Delta0block0;
   double d2_u1_dx = (-(5.0/2.0)*u1_B0_c - (1.0/12.0)*u1_B0_xm2 - (1.0/12.0)*u1_B0_xp2 + ((4.0/3.0))*u1_B0_xp1
     + ((4.0/3.0))*u1_B0_xm1)*inv2Delta0block0;
   double d2_u2_dx = (-(5.0/2.0)*u2_B0_c - (1.0/12.0)*u2_B0_xm2 - (1.0/12.0)*u2_B0_xp2 + ((4.0/3.0))*u2_B0_xp1
     + ((4.0/3.0))*u2_B0_xm1)*inv2Delta0block0;

   double d2_T_dy = (-(5.0/2.0)*T_B0_c - (1.0/12.0)*T_B0_ym2 - (1.0/12.0)*T_B0_yp2 + ((4.0/3.0))*T_B0_yp1 +
     ((4.0/3.0))*T_B0_ym1)*inv2Delta1block0;
   double d2_u0_dy = (-(5.0/2.0)*u0_B0_c - (1.0/12.0)*u0_B0_ym2 - (1.0/12.0)*u0_B0_yp2 + ((4.0/3.0))*u0_B0_yp1
     + ((4.0/3.0))*u0_B0_ym1)*inv2Delta1block0;
   double d2_u1_dy = (-(5.0/2.0)*u1_B0_c - (1.0/12.0)*u1_B0_ym2 - (1.0/12.0)*u1_B0_yp2 + ((4.0/3.0))*u1_B0_yp1
     + ((4.0/3.0))*u1_B0_ym1)*inv2Delta1block0;
   double d2_u2_dy = (-(5.0/2.0)*u2_B0_c - (1.0/12.0)*u2_B0_ym2 - (1.0/12.0)*u2_B0_yp2 + ((4.0/3.0))*u2_B0_yp1
     + ((4.0/3.0))*u2_B0_ym1)*inv2Delta1block0;

   double d1_wk0_dy = (-(2.0/3.0)*wk0_B0_ym1 - (1.0/12.0)*wk0_B0_yp2 + ((1.0/12.0))*wk0_B0_ym2 +
     ((2.0/3.0))*wk0_B0_yp1)*invDelta1block0;
   double d1_wk1_dy = (-(2.0/3.0)*wk1_B0_ym1 - (1.0/12.0)*wk1_B0_yp2 + ((1.0/12.0))*wk1_B0_ym2 +
     ((2.0/3.0))*wk1_B0_yp1)*invDelta1block0;

   double d2_T_dz = (-(5.0/2.0)*T_B0_c - (1.0/12.0)*T_B0_zm2 - (1.0/12.0)*T_B0_zp2 + ((4.0/3.0))*T_B0_zp1 +
     ((4.0/3.0))*T_B0_zm1)*inv2Delta2block0;
   double d2_u0_dz = (-(5.0/2.0)*u0_B0_c - (1.0/12.0)*u0_B0_zm2 - (1.0/12.0)*u0_B0_zp2 + ((4.0/3.0))*u0_B0_zp1
     + ((4.0/3.0))*u0_B0_zm1)*inv2Delta2block0;
   double d2_u1_dz = (-(5.0/2.0)*u1_B0_c - (1.0/12.0)*u1_B0_zm2 - (1.0/12.0)*u1_B0_zp2 + ((4.0/3.0))*u1_B0_zp1
     + ((4.0/3.0))*u1_B0_zm1)*inv2Delta2block0;
   double d2_u2_dz = (-(5.0/2.0)*u2_B0_c - (1.0/12.0)*u2_B0_zm2 - (1.0/12.0)*u2_B0_zp2 + ((4.0/3.0))*u2_B0_zp1
     + ((4.0/3.0))*u2_B0_zm1)*inv2Delta2block0;

   double d1_wk0_dz = (-(2.0/3.0)*wk0_B0_zm1 - (1.0/12.0)*wk0_B0_zp2 + ((1.0/12.0))*wk0_B0_zm2 +
     ((2.0/3.0))*wk0_B0_zp1)*invDelta2block0;
   double d1_wk2_dz = (-(2.0/3.0)*wk2_B0_zm1 - (1.0/12.0)*wk2_B0_zp2 + ((1.0/12.0))*wk2_B0_zm2 +
     ((2.0/3.0))*wk2_B0_zp1)*invDelta2block0;
   double d1_wk4_dz = (-(2.0/3.0)*wk4_B0_zm1 - (1.0/12.0)*wk4_B0_zp2 + ((1.0/12.0))*wk4_B0_zm2 +
     ((2.0/3.0))*wk4_B0_zp1)*invDelta2block0;
   double d1_wk5_dz = (-(2.0/3.0)*wk5_B0_zm1 - (1.0/12.0)*wk5_B0_zp2 + ((1.0/12.0))*wk5_B0_zm2 +
     ((2.0/3.0))*wk5_B0_zp1)*invDelta2block0;

   out->Residual1_B0 = 1.0*(((1.0/3.0))*d1_wk1_dy + ((1.0/3.0))*d1_wk2_dz + ((4.0/3.0))*d2_u0_dx + d2_u0_dy +
     d2_u0_dz)*invRe + Residual1_B0;

   out->Residual2_B0 = 1.0*(((1.0/3.0))*d1_wk0_dy + ((1.0/3.0))*d1_wk5_dz + ((4.0/3.0))*d2_u1_dy + d2_u1_dx +
     d2_u1_dz)*invRe + Residual2_B0;

   out->Residual3_B0 = 1.0*(((1.0/3.0))*d1_wk0_dz + ((1.0/3.0))*d1_wk4_dz + ((4.0/3.0))*d2_u2_dz + d2_u2_dx +
     d2_u2_dy)*invRe + Residual3_B0;

   out->Residual4_B0 = 1.0*(wk1_B0_c + wk3_B0)*invRe*wk1_B0_c + 1.0*(wk1_B0_c +
     wk3_B0)*invRe*wk3_B0 + 1.0*(wk2_B0_c + wk6_B0)*invRe*wk2_B0_c + 1.0*(wk2_B0_c
     + wk6_B0)*invRe*wk6_B0 + 1.0*(wk5_B0_c + wk7_B0)*invRe*wk5_B0_c +
     1.0*(wk5_B0_c + wk7_B0)*invRe*wk7_B0 + 1.0*(-(2.0/3.0)*wk0_B0_c - (2.0/3.0)*wk4_B0_c
     + ((4.0/3.0))*wk8_B0)*invRe*wk8_B0 + 1.0*(-(2.0/3.0)*wk0_B0_c - (2.0/3.0)*wk8_B0 +
     ((4.0/3.0))*wk4_B0_c)*invRe*wk4_B0_c + 1.0*(-(2.0/3.0)*wk4_B0_c - (2.0/3.0)*wk8_B0 +
     ((4.0/3.0))*wk0_B0_c)*invRe*wk0_B0_c + 1.0*(((1.0/3.0))*d1_wk0_dy + ((1.0/3.0))*d1_wk5_dz +
     ((4.0/3.0))*d2_u1_dy + d2_u1_dx + d2_u1_dz)*invRe*u1_B0_c + 1.0*(((1.0/3.0))*d1_wk0_dz +
     ((1.0/3.0))*d1_wk4_dz + ((4.0/3.0))*d2_u2_dz + d2_u2_dx + d2_u2_dy)*invRe*u2_B0_c +
     1.0*(((1.0/3.0))*d1_wk1_dy + ((1.0/3.0))*d1_wk2_dz + ((4.0/3.0))*d2_u0_dx + d2_u0_dy +
     d2_u0_dz)*invRe*u0_B0_c + 1.0*(d2_T_dx + d2_T_dy + d2_T_dz)*invPr*invRe*inv2Minf*inv_gamma_m1 +
     Residual4_B0;
}

struct opensbliblock00Kernel040_result {
  double rhoE_B0;
  double rhoE_RKold_B0;
  double rho_B0; 
  double rho_RKold_B0;
  double rhou0_B0;
  double rhou0_RKold_B0;
  double rhou1_B0;
  double rhou1_RKold_B0;
  double rhou2_B0;
  double rhou2_RKold_B0;
};

void opensbliblock00Kernel040(
    double Residual0_B0, double Residual1_B0, double Residual2_B0,
    double Residual3_B0, double Residual4_B0,
    double rhoE_B0, double rhoE_RKold_B0, double rho_B0, double rho_RKold_B0,
    double rhou0_B0, double rhou0_RKold_B0, double rhou1_B0, double rhou1_RKold_B0,
    double rhou2_B0, double rhou2_RKold_B0,
    const double *rkA, const double *rkB,
    opensbliblock00Kernel040_result *out) {
  out->rho_RKold_B0 = rkA[0]*rho_RKold_B0 + dt*Residual0_B0;
  out->rho_B0 = rkB[0]*out->rho_RKold_B0 + rho_B0;
  out->rhou0_RKold_B0 = rkA[0]*rhou0_RKold_B0 + dt*Residual1_B0;
  out->rhou0_B0 = rkB[0]*out->rhou0_RKold_B0 + rhou0_B0;
  out->rhou1_RKold_B0 = rkA[0]*rhou1_RKold_B0 + dt*Residual2_B0;
  out->rhou1_B0 = rkB[0]*out->rhou1_RKold_B0 + rhou1_B0;
  out->rhou2_RKold_B0 = rkA[0]*rhou2_RKold_B0 + dt*Residual3_B0;
  out->rhou2_B0 = rkB[0]*out->rhou2_RKold_B0 + rhou2_B0;
  out->rhoE_RKold_B0 = rkA[0]*rhoE_RKold_B0 + dt*Residual4_B0;
  out->rhoE_B0 = rkB[0]*out->rhoE_RKold_B0 + rhoE_B0;
}

struct opensbliblock00Kernel000_result {
  double rhoE_RKold_B0;
  double rho_RKold_B0;
  double rhou0_RKold_B0;
  double rhou1_RKold_B0;
  double rhou2_RKold_B0;
};

void opensbliblock00Kernel000(opensbliblock00Kernel000_result *out)
{
   out->rhoE_RKold_B0 = 0.0;
   out->rho_RKold_B0 = 0.0;
   out->rhou0_RKold_B0 = 0.0;
   out->rhou1_RKold_B0 = 0.0;
   out->rhou2_RKold_B0 = 0.0;
}

// stencil_0_55_00_00_23: 11 pts x, [-5..5]
struct opensbliblock00Kernel001_result {
  double rhoE_RKold_B0;
  double rho_RKold_B0;
  double rhou0_RKold_B0;
  double rhou1_RKold_B0;
  double rhou2_RKold_B0;
};

void opensbliblock00Kernel001(
    double rhoE_B0_xm5, double rhoE_B0_xm4, double rhoE_B0_xm3, double rhoE_B0_xm2, double rhoE_B0_xm1,
    double rhoE_B0_c, double rhoE_B0_xp1, double rhoE_B0_xp2, double rhoE_B0_xp3, double rhoE_B0_xp4, double rhoE_B0_xp5,
    double rho_B0_xm5, double rho_B0_xm4, double rho_B0_xm3, double rho_B0_xm2, double rho_B0_xm1,
    double rho_B0_c, double rho_B0_xp1, double rho_B0_xp2, double rho_B0_xp3, double rho_B0_xp4, double rho_B0_xp5,
    double rhou0_B0_xm5, double rhou0_B0_xm4, double rhou0_B0_xm3, double rhou0_B0_xm2, double rhou0_B0_xm1,
    double rhou0_B0_c, double rhou0_B0_xp1, double rhou0_B0_xp2, double rhou0_B0_xp3, double rhou0_B0_xp4, double rhou0_B0_xp5,
    double rhou1_B0_xm5, double rhou1_B0_xm4, double rhou1_B0_xm3, double rhou1_B0_xm2, double rhou1_B0_xm1,
    double rhou1_B0_c, double rhou1_B0_xp1, double rhou1_B0_xp2, double rhou1_B0_xp3, double rhou1_B0_xp4, double rhou1_B0_xp5,
    double rhou2_B0_xm5, double rhou2_B0_xm4, double rhou2_B0_xm3, double rhou2_B0_xm2, double rhou2_B0_xm1,
    double rhou2_B0_c, double rhou2_B0_xp1, double rhou2_B0_xp2, double rhou2_B0_xp3, double rhou2_B0_xp4, double rhou2_B0_xp5,
    opensbliblock00Kernel001_result *out)
{
    out->rho_RKold_B0 = 0.215044884112*rho_B0_c + 0.123755948787*rho_B0_xm2 + 0.123755948787*rho_B0_xp2 +
      0.018721609157*rho_B0_xm4 + 0.018721609157*rho_B0_xp4 - 0.059227575576*rho_B0_xm3 -
      0.059227575576*rho_B0_xp3 - 0.002999540835*rho_B0_xm5 - 0.002999540835*rho_B0_xp5 -
      0.187772883589*rho_B0_xp1 - 0.187772883589*rho_B0_xm1;

    out->rhou0_RKold_B0 = 0.215044884112*rhou0_B0_c + 0.123755948787*rhou0_B0_xm2 +
      0.123755948787*rhou0_B0_xp2 + 0.018721609157*rhou0_B0_xm4 + 0.018721609157*rhou0_B0_xp4 -
      0.059227575576*rhou0_B0_xm3 - 0.059227575576*rhou0_B0_xp3 - 0.002999540835*rhou0_B0_xm5 -
      0.002999540835*rhou0_B0_xp5 - 0.187772883589*rhou0_B0_xp1 - 0.187772883589*rhou0_B0_xm1;

    out->rhou1_RKold_B0 = 0.215044884112*rhou1_B0_c + 0.123755948787*rhou1_B0_xm2 +
      0.123755948787*rhou1_B0_xp2 + 0.018721609157*rhou1_B0_xm4 + 0.018721609157*rhou1_B0_xp4 -
      0.059227575576*rhou1_B0_xm3 - 0.059227575576*rhou1_B0_xp3 - 0.002999540835*rhou1_B0_xm5 -
      0.002999540835*rhou1_B0_xp5 - 0.187772883589*rhou1_B0_xp1 - 0.187772883589*rhou1_B0_xm1;

    out->rhou2_RKold_B0 = 0.215044884112*rhou2_B0_c + 0.123755948787*rhou2_B0_xm2 +
      0.123755948787*rhou2_B0_xp2 + 0.018721609157*rhou2_B0_xm4 + 0.018721609157*rhou2_B0_xp4 -
      0.059227575576*rhou2_B0_xm3 - 0.059227575576*rhou2_B0_xp3 - 0.002999540835*rhou2_B0_xm5 -
      0.002999540835*rhou2_B0_xp5 - 0.187772883589*rhou2_B0_xp1 - 0.187772883589*rhou2_B0_xm1;

    out->rhoE_RKold_B0 = 0.215044884112*rhoE_B0_c + 0.123755948787*rhoE_B0_xm2 +
      0.123755948787*rhoE_B0_xp2 + 0.018721609157*rhoE_B0_xm4 + 0.018721609157*rhoE_B0_xp4 -
      0.059227575576*rhoE_B0_xm3 - 0.059227575576*rhoE_B0_xp3 - 0.002999540835*rhoE_B0_xm5 -
      0.002999540835*rhoE_B0_xp5 - 0.187772883589*rhoE_B0_xp1 - 0.187772883589*rhoE_B0_xm1;
}

struct opensbliblock00Kernel002_result {
  double rhoE_B0;
  double rho_B0;
  double rhou0_B0;
  double rhou1_B0;
  double rhou2_B0;
};

void opensbliblock00Kernel002(
    double rhoE_RKold_B0, double rho_RKold_B0, double rhou0_RKold_B0, double rhou1_RKold_B0, double rhou2_RKold_B0,
    double rhoE_B0, double rho_B0, double rhou0_B0, double rhou1_B0, double rhou2_B0,
    opensbliblock00Kernel002_result *out)
{
   out->rho_B0 = -DRP_filt*rho_RKold_B0 + rho_B0;
   out->rhou0_B0 = -DRP_filt*rhou0_RKold_B0 + rhou0_B0;
   out->rhou1_B0 = -DRP_filt*rhou1_RKold_B0 + rhou1_B0;
   out->rhou2_B0 = -DRP_filt*rhou2_RKold_B0 + rhou2_B0;
   out->rhoE_B0 = -DRP_filt*rhoE_RKold_B0 + rhoE_B0;
}

// stencil_0_00_55_00_23: 11 pts y, [-5..5]
struct opensbliblock00Kernel003_result {
  double rhoE_RKold_B0;
  double rho_RKold_B0;
  double rhou0_RKold_B0;
  double rhou1_RKold_B0;
  double rhou2_RKold_B0;
};

void opensbliblock00Kernel003(
    double rhoE_B0_ym5, double rhoE_B0_ym4, double rhoE_B0_ym3, double rhoE_B0_ym2, double rhoE_B0_ym1,
    double rhoE_B0_c, double rhoE_B0_yp1, double rhoE_B0_yp2, double rhoE_B0_yp3, double rhoE_B0_yp4, double rhoE_B0_yp5,
    double rho_B0_ym5, double rho_B0_ym4, double rho_B0_ym3, double rho_B0_ym2, double rho_B0_ym1,
    double rho_B0_c, double rho_B0_yp1, double rho_B0_yp2, double rho_B0_yp3, double rho_B0_yp4, double rho_B0_yp5,
    double rhou0_B0_ym5, double rhou0_B0_ym4, double rhou0_B0_ym3, double rhou0_B0_ym2, double rhou0_B0_ym1,
    double rhou0_B0_c, double rhou0_B0_yp1, double rhou0_B0_yp2, double rhou0_B0_yp3, double rhou0_B0_yp4, double rhou0_B0_yp5,
    double rhou1_B0_ym5, double rhou1_B0_ym4, double rhou1_B0_ym3, double rhou1_B0_ym2, double rhou1_B0_ym1,
    double rhou1_B0_c, double rhou1_B0_yp1, double rhou1_B0_yp2, double rhou1_B0_yp3, double rhou1_B0_yp4, double rhou1_B0_yp5,
    double rhou2_B0_ym5, double rhou2_B0_ym4, double rhou2_B0_ym3, double rhou2_B0_ym2, double rhou2_B0_ym1,
    double rhou2_B0_c, double rhou2_B0_yp1, double rhou2_B0_yp2, double rhou2_B0_yp3, double rhou2_B0_yp4, double rhou2_B0_yp5,
    opensbliblock00Kernel003_result *out)
{
    out->rho_RKold_B0 = 0.215044884112*rho_B0_c + 0.123755948787*rho_B0_ym2 + 0.123755948787*rho_B0_yp2 +
      0.018721609157*rho_B0_ym4 + 0.018721609157*rho_B0_yp4 - 0.059227575576*rho_B0_ym3 -
      0.059227575576*rho_B0_yp3 - 0.002999540835*rho_B0_ym5 - 0.002999540835*rho_B0_yp5 -
      0.187772883589*rho_B0_yp1 - 0.187772883589*rho_B0_ym1;

    out->rhou0_RKold_B0 = 0.215044884112*rhou0_B0_c + 0.123755948787*rhou0_B0_ym2 +
      0.123755948787*rhou0_B0_yp2 + 0.018721609157*rhou0_B0_ym4 + 0.018721609157*rhou0_B0_yp4 -
      0.059227575576*rhou0_B0_ym3 - 0.059227575576*rhou0_B0_yp3 - 0.002999540835*rhou0_B0_ym5 -
      0.002999540835*rhou0_B0_yp5 - 0.187772883589*rhou0_B0_yp1 - 0.187772883589*rhou0_B0_ym1;

    out->rhou1_RKold_B0 = 0.215044884112*rhou1_B0_c + 0.123755948787*rhou1_B0_ym2 +
      0.123755948787*rhou1_B0_yp2 + 0.018721609157*rhou1_B0_ym4 + 0.018721609157*rhou1_B0_yp4 -
      0.059227575576*rhou1_B0_ym3 - 0.059227575576*rhou1_B0_yp3 - 0.002999540835*rhou1_B0_ym5 -
      0.002999540835*rhou1_B0_yp5 - 0.187772883589*rhou1_B0_yp1 - 0.187772883589*rhou1_B0_ym1;

    out->rhou2_RKold_B0 = 0.215044884112*rhou2_B0_c + 0.123755948787*rhou2_B0_ym2 +
      0.123755948787*rhou2_B0_yp2 + 0.018721609157*rhou2_B0_ym4 + 0.018721609157*rhou2_B0_yp4 -
      0.059227575576*rhou2_B0_ym3 - 0.059227575576*rhou2_B0_yp3 - 0.002999540835*rhou2_B0_ym5 -
      0.002999540835*rhou2_B0_yp5 - 0.187772883589*rhou2_B0_yp1 - 0.187772883589*rhou2_B0_ym1;

    out->rhoE_RKold_B0 = 0.215044884112*rhoE_B0_c + 0.123755948787*rhoE_B0_ym2 +
      0.123755948787*rhoE_B0_yp2 + 0.018721609157*rhoE_B0_ym4 + 0.018721609157*rhoE_B0_yp4 -
      0.059227575576*rhoE_B0_ym3 - 0.059227575576*rhoE_B0_yp3 - 0.002999540835*rhoE_B0_ym5 -
      0.002999540835*rhoE_B0_yp5 - 0.187772883589*rhoE_B0_yp1 - 0.187772883589*rhoE_B0_ym1;
}

struct opensbliblock00Kernel004_result {
  double rhoE_B0;
  double rho_B0;
  double rhou0_B0;
  double rhou1_B0;
  double rhou2_B0;
};

void opensbliblock00Kernel004(
    double rhoE_RKold_B0, double rho_RKold_B0, double rhou0_RKold_B0, double rhou1_RKold_B0, double rhou2_RKold_B0,
    double rhoE_B0, double rho_B0, double rhou0_B0, double rhou1_B0, double rhou2_B0,
    opensbliblock00Kernel004_result *out)
{
   out->rho_B0 = -DRP_filt*rho_RKold_B0 + rho_B0;
   out->rhou0_B0 = -DRP_filt*rhou0_RKold_B0 + rhou0_B0;
   out->rhou1_B0 = -DRP_filt*rhou1_RKold_B0 + rhou1_B0;
   out->rhou2_B0 = -DRP_filt*rhou2_RKold_B0 + rhou2_B0;
   out->rhoE_B0 = -DRP_filt*rhoE_RKold_B0 + rhoE_B0;
}

// stencil_0_00_00_55_23: 11 pts z, [-5..5]
struct opensbliblock00Kernel005_result {
  double rhoE_RKold_B0;
  double rho_RKold_B0;
  double rhou0_RKold_B0;
  double rhou1_RKold_B0;
  double rhou2_RKold_B0;
};

void opensbliblock00Kernel005(
    double rhoE_B0_zm5, double rhoE_B0_zm4, double rhoE_B0_zm3, double rhoE_B0_zm2, double rhoE_B0_zm1,
    double rhoE_B0_c, double rhoE_B0_zp1, double rhoE_B0_zp2, double rhoE_B0_zp3, double rhoE_B0_zp4, double rhoE_B0_zp5,
    double rho_B0_zm5, double rho_B0_zm4, double rho_B0_zm3, double rho_B0_zm2, double rho_B0_zm1,
    double rho_B0_c, double rho_B0_zp1, double rho_B0_zp2, double rho_B0_zp3, double rho_B0_zp4, double rho_B0_zp5,
    double rhou0_B0_zm5, double rhou0_B0_zm4, double rhou0_B0_zm3, double rhou0_B0_zm2, double rhou0_B0_zm1,
    double rhou0_B0_c, double rhou0_B0_zp1, double rhou0_B0_zp2, double rhou0_B0_zp3, double rhou0_B0_zp4, double rhou0_B0_zp5,
    double rhou1_B0_zm5, double rhou1_B0_zm4, double rhou1_B0_zm3, double rhou1_B0_zm2, double rhou1_B0_zm1,
    double rhou1_B0_c, double rhou1_B0_zp1, double rhou1_B0_zp2, double rhou1_B0_zp3, double rhou1_B0_zp4, double rhou1_B0_zp5,
    double rhou2_B0_zm5, double rhou2_B0_zm4, double rhou2_B0_zm3, double rhou2_B0_zm2, double rhou2_B0_zm1,
    double rhou2_B0_c, double rhou2_B0_zp1, double rhou2_B0_zp2, double rhou2_B0_zp3, double rhou2_B0_zp4, double rhou2_B0_zp5,
    opensbliblock00Kernel005_result *out)
{
    out->rho_RKold_B0 = 0.215044884112*rho_B0_c + 0.123755948787*rho_B0_zm2 + 0.123755948787*rho_B0_zp2 +
      0.018721609157*rho_B0_zm4 + 0.018721609157*rho_B0_zp4 - 0.059227575576*rho_B0_zm3 -
      0.059227575576*rho_B0_zp3 - 0.002999540835*rho_B0_zm5 - 0.002999540835*rho_B0_zp5 -
      0.187772883589*rho_B0_zp1 - 0.187772883589*rho_B0_zm1;

    out->rhou0_RKold_B0 = 0.215044884112*rhou0_B0_c + 0.123755948787*rhou0_B0_zm2 +
      0.123755948787*rhou0_B0_zp2 + 0.018721609157*rhou0_B0_zm4 + 0.018721609157*rhou0_B0_zp4 -
      0.059227575576*rhou0_B0_zm3 - 0.059227575576*rhou0_B0_zp3 - 0.002999540835*rhou0_B0_zm5 -
      0.002999540835*rhou0_B0_zp5 - 0.187772883589*rhou0_B0_zp1 - 0.187772883589*rhou0_B0_zm1;

    out->rhou1_RKold_B0 = 0.215044884112*rhou1_B0_c + 0.123755948787*rhou1_B0_zm2 +
      0.123755948787*rhou1_B0_zp2 + 0.018721609157*rhou1_B0_zm4 + 0.018721609157*rhou1_B0_zp4 -
      0.059227575576*rhou1_B0_zm3 - 0.059227575576*rhou1_B0_zp3 - 0.002999540835*rhou1_B0_zm5 -
      0.002999540835*rhou1_B0_zp5 - 0.187772883589*rhou1_B0_zp1 - 0.187772883589*rhou1_B0_zm1;

    out->rhou2_RKold_B0 = 0.215044884112*rhou2_B0_c + 0.123755948787*rhou2_B0_zm2 +
      0.123755948787*rhou2_B0_zp2 + 0.018721609157*rhou2_B0_zm4 + 0.018721609157*rhou2_B0_zp4 -
      0.059227575576*rhou2_B0_zm3 - 0.059227575576*rhou2_B0_zp3 - 0.002999540835*rhou2_B0_zm5 -
      0.002999540835*rhou2_B0_zp5 - 0.187772883589*rhou2_B0_zp1 - 0.187772883589*rhou2_B0_zm1;

    out->rhoE_RKold_B0 = 0.215044884112*rhoE_B0_c + 0.123755948787*rhoE_B0_zm2 +
      0.123755948787*rhoE_B0_zp2 + 0.018721609157*rhoE_B0_zm4 + 0.018721609157*rhoE_B0_zp4 -
      0.059227575576*rhoE_B0_zm3 - 0.059227575576*rhoE_B0_zp3 - 0.002999540835*rhoE_B0_zm5 -
      0.002999540835*rhoE_B0_zp5 - 0.187772883589*rhoE_B0_zp1 - 0.187772883589*rhoE_B0_zm1;
}

struct opensbliblock00Kernel006_result {
  double rhoE_B0;
  double rho_B0;
  double rhou0_B0;
  double rhou1_B0;
  double rhou2_B0;
};

void opensbliblock00Kernel006(
    double rhoE_RKold_B0, double rho_RKold_B0, double rhou0_RKold_B0, double rhou1_RKold_B0, double rhou2_RKold_B0,
    double rhoE_B0, double rho_B0, double rhou0_B0, double rhou1_B0, double rhou2_B0,
    opensbliblock00Kernel006_result *out)
{
   out->rho_B0 = -DRP_filt*rho_RKold_B0 + rho_B0;
   out->rhou0_B0 = -DRP_filt*rhou0_RKold_B0 + rhou0_B0;
   out->rhou1_B0 = -DRP_filt*rhou1_RKold_B0 + rhou1_B0;
   out->rhou2_B0 = -DRP_filt*rhou2_RKold_B0 + rhou2_B0;
   out->rhoE_B0 = -DRP_filt*rhoE_RKold_B0 + rhoE_B0;
}

#endif // OPENSBLIBLOCK00_KERNEL_H
