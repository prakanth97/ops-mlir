// A kernel that uses a name the translation unit does not declare is refused: the compiler recovered from the error
// with an expression the importer would otherwise misread.
// REQUIRES: ops-headers
// RUN: not ops-kernel-import --strict --kernel undeclared_name --source %s --ndim 2 -o - --manifest %t.json -- -I%ops_include 2>&1 | FileCheck %s
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.json

#include <ops_lib_core.h>

void undeclared_name(const ACC<double> &a, ACC<double> &b) { b(0, 0) = a(0, 0) * undeclared_factor; }

// CHECK: error: use of undeclared identifier 'undeclared_factor'
// CHECK: undeclared_name: not imported: the body of the kernel has unresolved names
// MANIFEST: "reason": "the body of the kernel has unresolved names
// MANIFEST: "status": "failed"
