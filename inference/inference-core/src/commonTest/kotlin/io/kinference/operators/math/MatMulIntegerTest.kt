package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MatMulIntegerTest {
    private fun getTargetPath(dirName: String) = "matmul_integer/$dirName/"

    @Test
    fun test_matmul_integer() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger"))
    }

    @Test
    fun test_matmul_integer_op() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger_op"))
    }
}
