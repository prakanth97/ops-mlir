#===============================================================================
# Applications
#===============================================================================

add_subdirectory(c)

if(OPS_ENABLE_FORTRAN AND OPS_BUILD_FORTRAN_APPS)
  add_subdirectory(fortran)
endif()
