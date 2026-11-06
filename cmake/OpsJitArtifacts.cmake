# ops_add_jit_artifacts(<target> KERNELS_JSON <jit_kernels.json>
#                       [IMPORTER <ops-kernel-import>] [OUTPUT_DIR <dir>] [CLANG_ARGS <flags>...])
#
# Imports every kernel listed in the translator's jit_kernels.json (OPS translator --jit) with ops-kernel-import and
# adds the generated ops_jit_artifacts.cpp to <target>. That file defines
#
#     const ops_jit_artifact ops_jit_artifact_<kernel> = {...};
#
# for every kernel (see ops_jit_artifact.h in OPS). A kernel that cannot be imported gets status 1 with the reason in
# its manifest: the loops that call it run the ahead-of-time code.
#
# The importer sees the compiler flags of <target> (INCLUDE_DIRECTORIES and COMPILE_DEFINITIONS, so they have to be
# set before the call or be generator expressions), plus CLANG_ARGS. An artifact is imported again when a file its
# translation unit read changes (depfile written by the importer; Makefile and Ninja generators).

set(_OPS_JIT_ARTIFACTS_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(ops_add_jit_artifacts target)
  cmake_parse_arguments(ARG "" "KERNELS_JSON;IMPORTER;OUTPUT_DIR" "CLANG_ARGS" ${ARGN})
  if(NOT ARG_KERNELS_JSON)
    message(FATAL_ERROR "ops_add_jit_artifacts: KERNELS_JSON is required")
  endif()
  if(NOT ARG_IMPORTER)
    if(TARGET ops-kernel-import)
      set(ARG_IMPORTER ops-kernel-import)
    else()
      find_program(ARG_IMPORTER ops-kernel-import REQUIRED)
    endif()
  endif()
  if(NOT ARG_OUTPUT_DIR)
    set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${target}_jit")
  endif()
  file(MAKE_DIRECTORY "${ARG_OUTPUT_DIR}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARG_KERNELS_JSON}")

  file(READ "${ARG_KERNELS_JSON}" json)
  string(JSON count LENGTH "${json}" kernels)

  # The flags of the target: one per line, so that no quoting is needed.
  set(inc "$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>")
  set(def "$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>")
  string(JOIN "\n" extra ${ARG_CLANG_ARGS})
  set(args_file "${ARG_OUTPUT_DIR}/clang_args.txt")
  file(GENERATE OUTPUT "${args_file}" CONTENT
       "$<$<BOOL:${inc}>:-I$<JOIN:${inc},\n-I>\n>$<$<BOOL:${def}>:-D$<JOIN:${def},\n-D>\n>${extra}\n")

  set(manifests "")
  set(kernel_names "")
  if(count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
      string(JSON k GET "${json}" kernels ${i} kernel)
      string(JSON source GET "${json}" kernels ${i} source)
      string(JSON ndim GET "${json}" kernels ${i} ndim)
      set(kinds "")
      string(JSON nk ERROR_VARIABLE kerr LENGTH "${json}" kernels ${i} kinds)
      if(NOT kerr AND nk GREATER 0)
        math(EXPR klast "${nk} - 1")
        foreach(j RANGE ${klast})
          string(JSON kind GET "${json}" kernels ${i} kinds ${j})
          list(APPEND kinds "${kind}")
        endforeach()
      endif()
      string(JOIN "," kinds_arg ${kinds})
      set(kinds_opt "")
      if(kinds_arg)
        set(kinds_opt --kinds ${kinds_arg})
      endif()

      set(bin "${ARG_OUTPUT_DIR}/${k}.mlirbc")
      set(man "${ARG_OUTPUT_DIR}/${k}.json")
      set(dep "${ARG_OUTPUT_DIR}/${k}.d")
      add_custom_command(
        OUTPUT "${man}"
        BYPRODUCTS "${bin}"
        COMMAND ${CMAKE_COMMAND} -E remove -f "${bin}"
        COMMAND ${ARG_IMPORTER} --kernel ${k} --source "${source}" --ndim ${ndim} ${kinds_opt}
                -o "${bin}" --manifest "${man}" --depfile "${dep}" --args-file "${args_file}"
        DEPENDS "${args_file}" "${source}" ${ARG_IMPORTER}
        DEPFILE "${dep}"
        COMMENT "Importing OPS kernel ${k}"
        VERBATIM)
      list(APPEND manifests "${man}")
      list(APPEND kernel_names "${k}")
    endforeach()
  endif()

  set(generated "${ARG_OUTPUT_DIR}/ops_jit_artifacts.cpp")
  add_custom_command(
    OUTPUT "${generated}"
    COMMAND ${CMAKE_COMMAND} -DDIR=${ARG_OUTPUT_DIR} "-DKERNELS=${kernel_names}" -DOUT=${generated}
            -P "${_OPS_JIT_ARTIFACTS_DIR}/OpsJitEmbed.cmake"
    DEPENDS ${manifests} "${_OPS_JIT_ARTIFACTS_DIR}/OpsJitEmbed.cmake"
    COMMENT "Embedding the JIT artifacts of ${target}"
    VERBATIM)
  target_sources(${target} PRIVATE "${generated}")
  target_include_directories(${target} PRIVATE "${ARG_OUTPUT_DIR}")
endfunction()
