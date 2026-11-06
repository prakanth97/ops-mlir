# Inputs of the kernel importer tests

Real kernels, copied verbatim (`scripts/make_importer_inputs.sh` regenerates them):

| File | Source |
|---|---|
| `cloverleaf_*_kernel.h` | `OPS/apps/c/CloverLeaf/*_kernel.h` |
| `cloverleaf3d_PdV_kernel.h` | `OPS/apps/c/CloverLeaf_3D/PdV_kernel.h` (pointer style, `OPS_ACCn`) |
| `tealeaf_set_field_kernel.h` | `OPS/apps/c/TeaLeaf/set_field_kernel.h` |
| `laplace2d_kernels.h` | `OPS/apps/c/laplace2d_tutorial/step7/laplace_kernels.h` |
| `multidim_kernel.h` | `OPS/apps/c/multiDim/multidim_kernel.h` |
| `tgv_dp_kernels.h`, `tgv_spdp_kernels.h` | functions of `reguly/tests` `apps/TGV_mixed/TGsym_DP` and `TGsym_SPDP` `opensbliblock00_kernels.h` |

`definitions.h` is a stand-in for the application header that some CloverLeaf kernels include.
