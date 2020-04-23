# Developer Overview

This guide provides an overview of IREE's project structure and main tools for
developers.

## Project Code Layout

[iree/](https://github.com/google/iree/blob/master/iree/)

*   Core IREE project

[integrations/](https://github.com/google/iree/blob/master/integrations/)

*   Integrations between IREE and other frameworks, such as TensorFlow

[bindings/](https://github.com/google/iree/blob/master/bindings/)

*   Language and platform bindings, such as Python

[colab/](https://github.com/google/iree/blob/master/colab/)

*   Colab notebooks for interactively using IREE's Python bindings

## IREE Code Layout

[iree/base/](https://github.com/google/iree/blob/master/iree/base/)

*   Common types and utilities used throughout IREE

[iree/compiler/](https://github.com/google/iree/blob/master/iree/compiler/)

*   IREE's MLIR dialects, LLVM compiler passes, module translation code, etc.

[iree/hal/](https://github.com/google/iree/blob/master/iree/hal/)

*   **H**ardware **A**bstraction **L**ayer for IREE's runtime, with
    implementations for hardware and software backends

[iree/schemas/](https://github.com/google/iree/blob/master/iree/schemas/)

*   Shared data storage format definitions, primarily using
    [FlatBuffers](https://google.github.io/flatbuffers/)

[iree/tools/](https://github.com/google/iree/blob/master/iree/tools/)

*   Assorted tools used to optimize, translate, and evaluate IREE

[iree/vm/](https://github.com/google/iree/blob/master/iree/vm/)

*   Bytecode **V**irtual **M**achine used to work with IREE modules and invoke
    IREE functions

## Developer Tools

IREE's compiler components accept programs and code fragments in several
formats, including high level TensorFlow Python code, serialized TensorFlow
[SavedModel](https://www.tensorflow.org/guide/saved_model) programs, and lower
level textual MLIR files using combinations of supported dialects like `xla_hlo`
and IREE's internal dialects. While input programs are ultimately compiled down
to modules suitable for running on some combination of IREE's target deployment
platforms, IREE's developer tools can run individual compiler passes,
translations, and other transformations step by step.

### iree-opt

`iree-opt` is a tool for testing IREE's compiler passes. It is similar to
[mlir-opt](https://github.com/llvm/llvm-project/tree/master/mlir/tools/mlir-opt)
and runs sets of IREE's compiler passes on `.mlir` input files. See "conversion"
in [MLIR's Glossary](https://mlir.llvm.org/getting_started/Glossary/#conversion)
for more information.

Test `.mlir` files that are checked in typically include a `RUN` block at the
top of the file that specifies which passes should be performed and if
`FileCheck` should be used to test the generated output.

For example, to run some passes on the
[reshape.mlir](https://github.com/google/iree/blob/master/iree/compiler/Translation/SPIRV/XLAToSPIRV/test/reshape.mlir)
test file:

```shell
$ bazel run iree/tools:iree-opt -- \
  -split-input-file \
  -iree-index-computation \
  -simplify-spirv-affine-exprs=false \
  -convert-iree-to-spirv \
  -verify-diagnostics \
  $PWD/iree/compiler/Translation/SPIRV/XLAToSPIRV/test/reshape.mlir
```

Custom passes may also be layered on top of `iree-opt`, see
[iree/samples/custom_modules/dialect](https://github.com/google/iree/blob/master/iree/samples/custom_modules/dialect)
for a sample.

### iree-translate

`iree-translate` converts MLIR input into external formats like IREE modules. It
is similar to
[mlir-translate](https://github.com/llvm/llvm-project/tree/master/mlir/tools/mlir-translate),
see "translation" in
[MLIR's Glossary](https://mlir.llvm.org/getting_started/Glossary/#translation)
for more information.

For example, to translate `simple.mlir` to an IREE module:

```shell
$ bazel run iree/tools:iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vmla \
  $PWD/iree/tools/test/simple.mlir \
  -o /tmp/simple.module
```

Custom translations may also be layered on top of `iree-translate`, see
[iree/samples/custom_modules/dialect](https://github.com/google/iree/blob/master/iree/samples/custom_modules/dialect)
for a sample.

### iree-run-module

The `iree-run-module` program takes an already translated IREE module as input
and executes an exported main function using the provided inputs.

This program can be used in sequence with `iree-translate` to translate a
`.mlir` file to an IREE module and then execute it. Here is an example command
that executes the simple `simple.module` compiled from `simple.mlir` above on
IREE's VMLA driver:

```shell
$ bazel run iree/tools:iree-run-module -- \
  --input_file=/tmp/simple.module \
  --driver=vmla \
  --entry_function=abs \
  --inputs="i32=-2"
```

### iree-check-module

The `iree-check-module` program takes an already translated IREE module as input
and executes it as a series of
[googletest](https://github.com/google/googletest) tests. This is the test
runner for the IREE [check framework](#runtime-tests).

```shell
$ bazel run iree/tools:iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vmla \
  $PWD/iree/test/e2e/xla_ops/abs.mlir \
  -o /tmp/abs.module
```

```shell
$ bazel run iree/modules/check:iree-check-module -- \
  /tmp/abs.module \
  --driver=vmla
```

### iree-run-mlir

The `iree-run-mlir` program takes a `.mlir` file as input, translates it to an
IREE bytecode module, and executes the module.

It is designed for testing and debugging, not production uses, and therefore
does some additional work that usually must be explicit, like marking every
function as exported by default and running all of them.

For example, to execute the contents of
[iree/tools/test/simple.mlir](https://github.com/google/iree/blob/master/iree/tools/test/simple.mlir):

```shell
$ bazel run iree/tools:iree-run-mlir -- \
  $PWD/iree/tools/test/simple.mlir \
  --input-value="i32=-2" \
  --iree-hal-target-backends=vmla
```

### iree-dump-module

The `iree-dump-module` program prints the contents of an IREE module FlatBuffer
file.

For example, to inspect the module translated above:

```shell
$ bazel run iree/tools:iree-dump-module -- /tmp/simple.module
```

## Testing

### Compiler Tests

IREE compilation tests are written as lit tests in the same style as MLIR. They
should generally follow the
[MLIR testing guide](https://mlir.llvm.org/getting_started/TestingGuide/) with a
few differences:

1.  We use [`iree-opt`](#iree-opt), which registers IREE dialects and doesn't
    register some unnecessary core ones.
2.  We use
    [`IreeFileCheck`](https://github.com/google/iree/tree/master/iree/tools/IreeFileCheck.sh),
    a shell-script wrapper around FileCheck that passes it a few
    `--do-the-right-thing` flags.

As with all parts of the IREE compiler, these should not have a dependency on
the runtime.

### Runtime Tests

Note: IREE runtime tests historically used `iree-run-mlir`. We are in the
process of transitioning them to use `iree-check-module`, but that migration is
incomplete, so some tests still use `iree-run-mlir`.

IREE uses a custom framework (called `check`) for runtime tests. This is made up
of a few parts:

1.  The [`check` dialect](https://google.github.io/iree/Dialects/CheckDialect)
    that defines ops for test assertions.
2.  An IREE
    [native module](https://github.com/google/iree/tree/master/iree/modules/check/native_module.h)
    implementation of those ops using
    [GoogleTest](https://github.com/google/googletest).
3.  The `iree-check-module` [test runner](#iree-check-module) that runs a
    compiled IREE module as a GoogleTest test suite.
4.  A set of
    [Bazel](https://github.com/google/iree/tree/master/build_tools/bazel/iree_check_test.bzl)
    and
    [CMake](https://github.com/google/iree/tree/master/build_tools/cmake/iree_check_test.cmake)
    build rules that allow defining test targets using MLIR source files,
    compiling them to an IREE module on the host and running them as a test
    suite on the target.
5.  Compiler hint operations in the
    [IREE dialect](https://google.github.io/iree/Dialects/IREEDialect) that
    allow defining values that the compiler will not optimize away.

#### Writing a Test

An IREE check test consists of a `.mlir` source file that can be compiled with
`iree-translate`. It should contain an IREE module where each exported function
is nullary and void and corresponds to a single test case.

As an example, here are some tests for the XLA HLO floor operation:

```mlir
func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[0.0, 1.1, 2.5, 4.9]> : tensor<4xf32>
  %result = "xla_hlo.floor"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>): tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<101.3> : tensor<f32>
  %result = "xla_hlo.floor"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<101.0> : tensor<f32>): tensor<f32>
  return
}

func @double() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<11.2> : tensor<f64>
  %result = "xla_hlo.floor"(%input) : (tensor<f64>) -> tensor<f64>
  check.expect_almost_eq_const(%result, dense<11.0> : tensor<f64>): tensor<f64>
  return
}

func @negative() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<-1.1> : tensor<f32>
  %result = "xla_hlo.floor"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-2.0> : tensor<f32>): tensor<f32>
  return
}
```

Each of these exported functions will be used to create a test case in gtest.

Note the use of
[`iree.unfoldable_constant`](https://google.github.io/iree/Dialects/IREEDialect#ireeunfoldable_constant-ireeunfoldableconstantop)
to specify test constants. If we were to use a regular constant, the compiler
would "helpfully" fold away everything at compile time and our test would not
actually test the runtime. `unfoldable_constant` hides the value of the constant
from the compiler so it cannot use it at compile time. To hide an arbitrary
SSA-value, you can use
[`iree.do_not_optimize`](https://google.github.io/iree/Dialects/IREEDialect#ireedo_not_optimize-ireedonotoptimizeop).
This wraps any value in an unoptimizable identity function
(`unfoldable_constant` is implemented using `do_not_optimize`).

Next we use this input constant to exercise the runtime feature under test (in
this case, just a single floor operation). Finally, we use a check dialect
operation to make an assertion about the output. Here we use the
`expect_almost_eq_const` op: *almost* because we are comparing floats and want
to allow for floating-point imprecision, and *const* because we want to compare
it to a constant value. This last part is just syntactic sugar around

```mlir
%expected = constant dense<101.0> : tensor<f32>
check.expect_almost_eq(%result, %expected) : tensor<f32>
```

The output of running this test looks like:

```txt
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from module
[ RUN      ] module.tensor
[       OK ] module.tensor (76 ms)
[ RUN      ] module.scalar
[       OK ] module.scalar (79 ms)
[ RUN      ] module.double
[       OK ] module.double (55 ms)
[ RUN      ] module.negative
[       OK ] module.negative (54 ms)
[----------] 4 tests from module (264 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (264 ms total)
[  PASSED  ] 4 tests.
```

The "module" name for the test suite comes from the default name for an implicit
MLIR module. If there was an explicit named top-level module in this file, the
test suite would inherit its name.

#### Dynamic Shapes

Constants with dynamic shape are not yet supported. See
https://github.com/google/iree/issues/1601. For now, these tests have to use
`iree-run-mlir` lit tests and input arguments.

#### Build Rules

A single `.mlir` source file can be turned into a test target with the
`iree_check_test` Bazel macro (and corresponding CMake function).

```bzl
iree_check_test(
    name = "check_vmla_vmla_floor.mlir",
    src = "floor.mlir",
    driver = "vmla",
    target_backend = "vmla",
)
```

The target naming convention is "check_backend_driver_src". The generated test
will automatically be tagged with a "driver=vmla" tag, which can help filter
tests by backend (especially when many tests are generated, as below).

Usually we want to create a suite of tests across many backends and drivers.
This can be accomplished with additional macros. For a single backend/driver
pair:

```bzl
iree_check_single_backend_test_suite(
    name = "check_vmla_vmla",
    srcs = glob(["*.mlir"]),
    driver = "vmla",
    target_backend = "vmla",
)
```

This will generate a separate test target for each file in `srcs` with a name
following the convention above as well as a Bazel
[test_suite](https://docs.bazel.build/versions/master/be/general.html#test_suite)
called "check_vmla_vmla" that will run all the generated tests.

You can also generate suites across multiple pairs:

```bzl
iree_check_test_suite(
    name = "check",
    srcs = ["success.mlir"],
    # Leave this argument off to run on all supported backend/driver pairs.
    target_backends_and_drivers = [
        ("vmla", "vmla"),
        ("vulkan-spirv", "vulkan"),
    ],
)
```

This will create a test per source file and backend/driver pair, a test suite
per backend/driver pair, and a test suite, "check", that will run all the tests.

The CMake functions follow a similar pattern. The calls to them are generated in
our `CMakeLists.txt` file by
[bazel_to_cmake](https://github.com/google/iree/tree/master/build_tools/bazel_to_cmake/bazel_to_cmake.py).
