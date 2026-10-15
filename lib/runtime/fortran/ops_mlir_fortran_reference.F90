!===- ops_mlir_fortran_reference.F90 - OPS Fortran JIT capture ----------===!
!
! Part of OPS-MLIR Project
!
! This file is distributed under the MIT License.
! See LICENSE.txt for details.
!
! Drop-in replacement for the `OPS_Fortran_Reference` module that the OPS
! Fortran translator normally generates per application. Instead of
! rewriting every `call ops_par_loop(...)` into a generated `<kernel>_host`
! stub, this module implements `ops_par_loop` directly as a generic
! interface that captures the loop and hands it to the OPS-MLIR JIT engine.
! An application therefore only changes its `use` line:
!
!     use OPS_Fortran_Reference   ->   use OPS_MLIR_Fortran_Reference
!
! Everything else in the application -- ops_decl_block, ops_decl_dat,
! ops_arg_dat, ops_arg_idx, ops_arg_reduce, ... -- is re-exported unchanged
! from OPS_Fortran_Declarations, so no OPS translator step is involved.
!
!===----------------------------------------------------------------------===!

module OPS_MLIR_Fortran_Reference

    use OPS_Fortran_Declarations
    use OPS_Fortran_RT_Support
    use, intrinsic :: ISO_C_BINDING

    implicit none

    ! Deliberately no `private` statement: entities pulled in from
    ! OPS_Fortran_Declarations / OPS_Fortran_RT_Support above are re-exported
    ! so applications keep using the ordinary OPS Fortran API.

    !=======================================================================
    ! ops_par_loop
    !=======================================================================
    ! OPS Fortran's ops_par_loop already carries the kernel *name* as a
    ! string argument, so a capture layer can recover the kernel identity
    ! without a source rewriter or a generated manifest. The name is what
    ! the JIT later looks up in the kernel module it compiles.
    interface ops_par_loop
        module procedure ops_mlir_par_loop_1, ops_mlir_par_loop_2, &
                         ops_mlir_par_loop_3, ops_mlir_par_loop_4, &
                         ops_mlir_par_loop_5, ops_mlir_par_loop_6, &
                         ops_mlir_par_loop_7, ops_mlir_par_loop_8
    end interface ops_par_loop

    !=======================================================================
    ! C ABI implemented by lib/runtime/fortran/FortranCapture.cpp
    !=======================================================================
    interface

        subroutine ops_mlir_fortran_enqueue_c(kernel_name, block, dims, range, &
                                              args, nargs) &
                bind(C, name='ops_mlir_fortran_enqueue')
            use, intrinsic :: ISO_C_BINDING
            character(kind=c_char)     :: kernel_name(*)
            type(c_ptr), value         :: block
            integer(kind=c_int), value :: dims
            type(c_ptr), value         :: range
            type(c_ptr), value         :: args
            integer(kind=c_int), value :: nargs
        end subroutine ops_mlir_fortran_enqueue_c

        subroutine ops_mlir_fortran_compile_and_execute_c() &
                bind(C, name='ops_mlir_fortran_compile_and_execute')
        end subroutine ops_mlir_fortran_compile_and_execute_c

        subroutine ops_mlir_fortran_set_kernel_source_file_c(path) &
                bind(C, name='ops_mlir_fortran_set_kernel_source_file')
            use, intrinsic :: ISO_C_BINDING
            character(kind=c_char) :: path(*)
        end subroutine ops_mlir_fortran_set_kernel_source_file_c

        subroutine ops_mlir_fortran_add_kernel_dependency_c(path) &
                bind(C, name='ops_mlir_fortran_add_kernel_dependency')
            use, intrinsic :: ISO_C_BINDING
            character(kind=c_char) :: path(*)
        end subroutine ops_mlir_fortran_add_kernel_dependency_c

        subroutine ops_mlir_fortran_register_constant_c(name, data, bytes) &
                bind(C, name='ops_mlir_fortran_register_constant')
            use, intrinsic :: ISO_C_BINDING
            character(kind=c_char)     :: name(*)
            type(c_ptr), value         :: data
            integer(kind=c_int), value :: bytes
        end subroutine ops_mlir_fortran_register_constant_c

        subroutine ops_mlir_fortran_sync_all_host_buffers_c() &
                bind(C, name='ops_mlir_fortran_sync_all_host_buffers')
        end subroutine ops_mlir_fortran_sync_all_host_buffers_c

        subroutine ops_mlir_fortran_exit_c() &
                bind(C, name='ops_mlir_fortran_exit')
        end subroutine ops_mlir_fortran_exit_c

    end interface

    interface ops_mlir_register_kernel_constant
        module procedure ops_mlir_register_kernel_constant_r8, &
                         ops_mlir_register_kernel_constant_i4
    end interface ops_mlir_register_kernel_constant

    !=======================================================================
    ! ops_decl_const
    !=======================================================================
    ! OPS_Fortran_Declarations only exposes a generic ops_decl_const under
    ! OPS_F2C_INTEROP, and that path binds to `ops_decl_const_f2c`, which the
    ! OPS C library does not define -- it comes from translator-generated
    ! f2c glue. ops-mlir has no translator step, so the capture module
    ! provides ops_decl_const itself under the same name and signature the
    ! application already uses. Registering the value with the JIT is all
    ! that is needed: constants are baked into kernel bodies at compile time
    ! rather than read from OPS at run time.
    interface ops_decl_const
        module procedure ops_mlir_decl_const_r8, ops_mlir_decl_const_i4
    end interface ops_decl_const

contains

    !=======================================================================
    ! Driver-facing helpers
    !=======================================================================

    !> Point the JIT at the module holding the application's kernel
    !> procedures. This is the Fortran counterpart of
    !> set_kernel_source_file() in the C++ wrapper.
    subroutine ops_mlir_set_kernel_source_file(path)
        character(len=*), intent(in) :: path
        call ops_mlir_fortran_set_kernel_source_file_c(trim(path)//C_NULL_CHAR)
    end subroutine ops_mlir_set_kernel_source_file

    !> Register a module the kernel module `use`s, such as the application's
    !> constants. The JIT compiles it first so the kernels can be checked
    !> against it; register a module before the ones that use it.
    subroutine ops_mlir_add_kernel_dependency(path)
        character(len=*), intent(in) :: path
        call ops_mlir_fortran_add_kernel_dependency_c(trim(path)//C_NULL_CHAR)
    end subroutine ops_mlir_add_kernel_dependency

    !> Compile and run everything captured since the last call.
    subroutine ops_mlir_compile_and_execute()
        call ops_mlir_fortran_compile_and_execute_c()
    end subroutine ops_mlir_compile_and_execute

    subroutine ops_mlir_sync_all_host_buffers()
        call ops_mlir_fortran_sync_all_host_buffers_c()
    end subroutine ops_mlir_sync_all_host_buffers

    !> Tears the JIT's cached ExecutionEngines down before ops_exit runs.
    subroutine ops_mlir_exit()
        call ops_mlir_fortran_exit_c()
    end subroutine ops_mlir_exit

    !> Makes a module constant's value visible to the kernel translator,
    !> which bakes it into translated kernel bodies. The
    !> `name` must be the Fortran source name (e.g. "pi"); the translator
    !> matches it against the mangled FIR global symbol.
    subroutine ops_mlir_register_kernel_constant_r8(name, data)
        character(len=*), intent(in)     :: name
        real(kind=8), intent(in), target :: data
        call ops_mlir_fortran_register_constant_c(trim(name)//C_NULL_CHAR, &
                                                  c_loc(data), 8_c_int)
    end subroutine ops_mlir_register_kernel_constant_r8

    subroutine ops_mlir_register_kernel_constant_i4(name, data)
        character(len=*), intent(in)        :: name
        integer(kind=4), intent(in), target :: data
        call ops_mlir_fortran_register_constant_c(trim(name)//C_NULL_CHAR, &
                                                  c_loc(data), 4_c_int)
    end subroutine ops_mlir_register_kernel_constant_i4

    !> ops_decl_const, with the signature stock OPS Fortran applications use.
    !> `dim` and `typ` are accepted for source compatibility; the JIT takes
    !> the element type from the FIR global it is baking into, so only the
    !> value matters here.
    subroutine ops_mlir_decl_const_r8(name, dim, typ, data)
        character(len=*), intent(in)     :: name
        integer(kind=4), intent(in)      :: dim
        character(len=*), intent(in)     :: typ
        real(kind=8), intent(in), target :: data
        call ops_mlir_fortran_register_constant_c(trim(name)//C_NULL_CHAR, &
                                                  c_loc(data), 8_c_int * dim)
    end subroutine ops_mlir_decl_const_r8

    subroutine ops_mlir_decl_const_i4(name, dim, typ, data)
        character(len=*), intent(in)        :: name
        integer(kind=4), intent(in)         :: dim
        character(len=*), intent(in)        :: typ
        integer(kind=4), intent(in), target :: data
        call ops_mlir_fortran_register_constant_c(trim(name)//C_NULL_CHAR, &
                                                  c_loc(data), 4_c_int * dim)
    end subroutine ops_mlir_decl_const_i4

    !=======================================================================
    ! Range conversion
    !=======================================================================

    !> OPS Fortran iteration ranges are one-based and inclusive at both
    !> ends; the OPS C core wants zero-based, half-open ranges. This is the
    !> same `[lo-1, hi]` conversion the translator-generated host stubs
    !> apply before calling into the C runtime, so a captured LoopDesc is
    !> indistinguishable from one captured through the C++ wrapper.
    subroutine ops_mlir_convert_range(dims, range, converted)
        integer(kind=c_int), intent(in)  :: dims
        integer(kind=4), intent(in)      :: range(*)
        integer(kind=c_int), intent(out) :: converted(*)
        integer :: d
        do d = 1, dims
            converted(2*d-1) = range(2*d-1) - 1
            converted(2*d)   = range(2*d)
        end do
    end subroutine ops_mlir_convert_range

    !=======================================================================
    ! ops_par_loop implementations
    !=======================================================================
    ! `kernel` is declared `external` and its address is never taken: for
    ! Fortran the kernel body is always translated from source, so the JIT
    ! never needs a host function pointer to fall back on (the C++ path
    ! passes one only so CPU backends *can* bind the symbol). The kernel
    ! name string is the identity used from here on.

    subroutine ops_mlir_par_loop_1(kernel, name, block, dims, range, arg1)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1
        type(ops_arg), target          :: packed(1)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 1_c_int)
    end subroutine ops_mlir_par_loop_1

    subroutine ops_mlir_par_loop_2(kernel, name, block, dims, range, arg1, arg2)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1, arg2
        type(ops_arg), target          :: packed(2)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1; packed(2) = arg2
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 2_c_int)
    end subroutine ops_mlir_par_loop_2

    subroutine ops_mlir_par_loop_3(kernel, name, block, dims, range, arg1, arg2, arg3)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1, arg2, arg3
        type(ops_arg), target          :: packed(3)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1; packed(2) = arg2; packed(3) = arg3
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 3_c_int)
    end subroutine ops_mlir_par_loop_3

    subroutine ops_mlir_par_loop_4(kernel, name, block, dims, range, arg1, arg2, arg3, arg4)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1, arg2, arg3, arg4
        type(ops_arg), target          :: packed(4)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1; packed(2) = arg2; packed(3) = arg3; packed(4) = arg4
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 4_c_int)
    end subroutine ops_mlir_par_loop_4

    subroutine ops_mlir_par_loop_5(kernel, name, block, dims, range, arg1, arg2, arg3, arg4, arg5)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1, arg2, arg3, arg4, arg5
        type(ops_arg), target          :: packed(5)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1; packed(2) = arg2; packed(3) = arg3; packed(4) = arg4
        packed(5) = arg5
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 5_c_int)
    end subroutine ops_mlir_par_loop_5

    subroutine ops_mlir_par_loop_6(kernel, name, block, dims, range, arg1, arg2, arg3, arg4, arg5, arg6)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1, arg2, arg3, arg4, arg5, arg6
        type(ops_arg), target          :: packed(6)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1; packed(2) = arg2; packed(3) = arg3; packed(4) = arg4
        packed(5) = arg5; packed(6) = arg6
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 6_c_int)
    end subroutine ops_mlir_par_loop_6

    subroutine ops_mlir_par_loop_7(kernel, name, block, dims, range, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1, arg2, arg3, arg4, arg5, arg6, arg7
        type(ops_arg), target          :: packed(7)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1; packed(2) = arg2; packed(3) = arg3; packed(4) = arg4
        packed(5) = arg5; packed(6) = arg6; packed(7) = arg7
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 7_c_int)
    end subroutine ops_mlir_par_loop_7

    subroutine ops_mlir_par_loop_8(kernel, name, block, dims, range, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        external :: kernel
        character(len=*), intent(in)   :: name
        type(ops_block), intent(in)    :: block
        integer(kind=4), intent(in)    :: dims
        integer(kind=4), intent(in)    :: range(*)
        type(ops_arg), intent(in)      :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
        type(ops_arg), target          :: packed(8)
        integer(kind=c_int), target    :: converted(2*OPS_MAX_DIM)

        packed(1) = arg1; packed(2) = arg2; packed(3) = arg3; packed(4) = arg4
        packed(5) = arg5; packed(6) = arg6; packed(7) = arg7; packed(8) = arg8
        call ops_mlir_convert_range(int(dims, c_int), range, converted)
        call ops_mlir_fortran_enqueue_c(trim(name)//C_NULL_CHAR, &
            block%blockCptr, int(dims, c_int), c_loc(converted(1)), &
            c_loc(packed(1)), 8_c_int)
    end subroutine ops_mlir_par_loop_8

end module OPS_MLIR_Fortran_Reference
