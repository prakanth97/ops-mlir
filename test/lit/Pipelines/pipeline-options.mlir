// Named pipelines and their option strings.
// RUN: not ops-opt %s -ops-jit-cpu="bogus=1" 2>&1 | FileCheck %s --check-prefix=BOGUS
// RUN: not ops-opt %s -ops-jit-cpu="omp=maybe" 2>&1 | FileCheck %s --check-prefix=BADBOOL
// RUN: not ops-opt %s -ops-jit-cpu="specialize=everything" 2>&1 | FileCheck %s --check-prefix=BADMODE
// RUN: not ops-opt %s -ops-jit-cpu="loops=poly" 2>&1 | FileCheck %s --check-prefix=BADLOOPS
// RUN: not ops-opt %s -ops-jit-cpu="affine-passes=func.func(affine-loop-tile)" 2>&1 | FileCheck %s --check-prefix=NEEDAFFINE
// RUN: not ops-opt %s -ops-jit-cpu="loops=affine affine-passes=no-such-pass" 2>&1 | FileCheck %s --check-prefix=NOPASS
// RUN: not ops-opt %s -ops-jit-plan="placement=middle" 2>&1 | FileCheck %s --check-prefix=PLACEMENT
// RUN: not ops-opt %s -ops-jit-frontend="anything=1" 2>&1 | FileCheck %s --check-prefix=NOOPTS
// RUN: ops-opt %s -ops-jit-plan="max-loops=2 reorder=false" | FileCheck %s
// RUN: ops-opt %s -ops-jit-plan="max-loops=2,reorder=false" | FileCheck %s

// BOGUS: unknown option 'bogus' (known: omp, specialize, soa, fma, loops, affine-passes)
// BADBOOL: option 'omp' expects true or false, got 'maybe'
// BADMODE: option 'specialize' must be none, ranges or shapes, got 'everything'
// BADLOOPS: option 'loops' must be scf or affine, got 'poly'
// NEEDAFFINE: option 'affine-passes' needs loops=affine
// NOPASS: option 'affine-passes':
// PLACEMENT: option 'placement' must be earliest or latest, got 'middle'
// NOOPTS: unknown option 'anything' (there are none)
// CHECK: ops.chain
ops.chain @empty() {
  ops.return
}
