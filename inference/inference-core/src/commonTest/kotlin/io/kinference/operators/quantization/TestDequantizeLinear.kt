package io.kinference.operators.quantization

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TestDequantizeLinear {
    private fun getTargetPath(dirName: String) = "dequantize_linear/$dirName/"

    @Test
    fun test_linear_dequantization_defaults() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear"))
    }

    @Test
    fun test_linear_dequantization_per_tensor() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_tensor"))
    }

    @Test
    fun test_linear_dequantization_per_axis_1() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_axis_1"))
    }

    @Test
    fun test_linear_dequantization_per_axis() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_axis"))
    }
}
