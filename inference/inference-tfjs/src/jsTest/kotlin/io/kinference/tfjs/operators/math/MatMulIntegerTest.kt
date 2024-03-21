package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MatMulIntegerTest {
    private fun getTargetPath(dirName: String) = "matmul_integer/$dirName/"

    @Test
    fun test_matmul_integer() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger"))
    }

    @Test
    fun test_matmul_integer_op() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger_op"))
    }
}
