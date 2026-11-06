// Verbatim from TGV_mixed/TGsym_SPDP/opensbliblock00_kernels.h (reguly/tests, apps/TGV_mixed).
void opensbliblock00Kernel003(const ACC<float> &wk39_B0, ACC<float> &wk0_B0)
{
    wk0_B0(0,0,0) = (-(2.0f/3.0f)*wk39_B0(-1,0,0) - (1.0f/12.0f)*wk39_B0(2,0,0) + ((1.0f/12.0f))*wk39_B0(-2,0,0) +
      ((2.0f/3.0f))*wk39_B0(1,0,0))*invDelta0block0;

}

void opensbliblock00Kernel005(const ACC<double> &rhou0_B0, ACC<float> &wk1_B0)
{
    wk1_B0(0,0,0) = (-(2.0f/3.0f)*rhou0_B0(-1,0,0) - (1.0f/12.0f)*rhou0_B0(2,0,0) + ((1.0f/12.0f))*rhou0_B0(-2,0,0) +
      ((2.0f/3.0f))*rhou0_B0(1,0,0))*invDelta0block0;

}

