package io.kinference.operators.quantization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class DequantizeLinearTest {
    private fun getTargetPath(dirName: String) = "/dequantize_linear/$dirName/"

    @Test
    fun test_linear_dequantization_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear"))
    }

    @Test
    fun test_linear_dequantization_per_tensor()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_tensor"))
    }

    @Test
    fun test_linear_dequantization_per_axis_1()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_axis_1"))
    }

    @Test
    fun test_linear_dequantization_per_axis()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_axis"))
    }
}
