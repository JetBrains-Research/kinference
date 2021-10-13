package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MatMulIntegerTest {
    private fun getTargetPath(dirName: String) = "/matmul_integer/$dirName/"

    @Test
    fun test_matmul_integer() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger"))
    }

    @Test
    fun test_matmul_integer_op() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger_op"))
    }
}
