package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.runners.AccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class MatMulIntegerTestToFloatTest {
    private fun getTargetPath(dirName: String) = "matmul_integer_to_float/$dirName/"

    @Test
    fun test_matmul_integer_to_float()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmulintegertofloat"), AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_matmul_integer_to_float_op()  = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmulintegertofloat_op"), AccuracyRunner.QUANT_DELTA)
    }
}
