// Verbatim from TGV_mixed/TGsym_DP/opensbliblock00_kernels.h (reguly/tests, apps/TGV_mixed).
 void opensbliblock00Kernel087(ACC<double> &rhoE_B0, ACC<double> &rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0,
ACC<double> &rhou2_B0, const int *idx)
{
   double p = 0.0;
   double r = 0.0;
   double u0 = 0.0;
   double u1 = 0.0;
   double u2 = 0.0;
   double x0 = 0.0;
   double x1 = 0.0;
   double x2 = 0.0;
   x0 = Delta0block0*idx[0];

   x1 = Delta1block0*idx[1];

   x2 = Delta2block0*idx[2];

   u0 = cos(x1)*cos(x2)*sin(x0);

   u1 = -cos(x0)*cos(x2)*sin(x1);

   u2 = 0.0;

   p = (2.0 + cos(2.0*x2))*(0.0625*cos(2.0*x0) + 0.0625*cos(2.0*x1)) + 1.0/((Minf*Minf)*gama);

   r = gama*p*(Minf*Minf);

   rho_B0(0,0,0) = r;

   rhou0_B0(0,0,0) = r*u0;

   rhou1_B0(0,0,0) = r*u1;

   rhou2_B0(0,0,0) = r*u2;

   rhoE_B0(0,0,0) = p/(-1 + gama) + 0.5*r*((u0*u0) + (u1*u1) + (u2*u2));

}

 void opensbliblock00Kernel081(ACC<double> &rhoE_B0, ACC<double> &rho_B0, ACC<double> &rhou0_B0, ACC<double> &rhou1_B0,
ACC<double> &rhou2_B0)
{
   rho_B0(-1,0,0) = rho_B0(1,0,0);

   rhou0_B0(-1,0,0) = -1.0*rhou0_B0(1,0,0);

   rhou1_B0(-1,0,0) = rhou1_B0(1,0,0);

   rhou2_B0(-1,0,0) = rhou2_B0(1,0,0);

   rhoE_B0(-1,0,0) = rhoE_B0(1,0,0);

   rho_B0(-2,0,0) = rho_B0(2,0,0);

   rhou0_B0(-2,0,0) = -1.0*rhou0_B0(2,0,0);

   rhou1_B0(-2,0,0) = rhou1_B0(2,0,0);

   rhou2_B0(-2,0,0) = rhou2_B0(2,0,0);

   rhoE_B0(-2,0,0) = rhoE_B0(2,0,0);

}

void opensbliblock00Kernel002(const ACC<double> &rho_B0, const ACC<double> &rhou0_B0, ACC<double> &u0_B0)
{
   u0_B0(0,0,0) = rhou0_B0(0,0,0)/rho_B0(0,0,0);

}

