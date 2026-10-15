"""
Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design (see xdsl_impl/laplace_stencil_1.mlir
for the target IR shape).

Usage:
    python xdsl_impl/ops_to_xdsl.py xdsl_impl/laplace_sample.mlir
"""

import sys
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects import func, stencil
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import (
    ConvertStencilToLLMLIRPass,
)
from xdsl.passes import PassPipeline


sys.path.insert(0, str(Path(__file__).parent))
from ops_dialect import OPS
from ops_to_stencil import OPSToStencilPass

def convert_ir_text(text: str) -> str:
    """String-in/string-out entry point for embedding (see
    lib/runtime/common/JITEngine.cpp::runXdslLowering). Unlike main(), this never
    touches the filesystem or stdout/stderr -- the caller owns the IR
    string and the result string.
    """
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(OPS)
    ctx.load_dialect(stencil.Stencil)
    ctx.load_dialect(func.Func)

    module = Parser(ctx, text).parse_module()
    pipeline = PassPipeline([
        OPSToStencilPass(),
        ConvertStencilToLLMLIRPass(),
    ])

    pipeline.apply(ctx, module)
    module.verify()

    from io import StringIO

    out = StringIO()
    Printer(stream=out).print_op(module)
    return out.getvalue()


def main() -> None:
    args = sys.argv[1:]

    if len(args) != 1:
        print(f"usage: {sys.argv[0]} <input.mlir>", file=sys.stderr)
        sys.exit(1)

    text = Path(args[0]).read_text()
    print(convert_ir_text(text))


if __name__ == "__main__":
    main()
