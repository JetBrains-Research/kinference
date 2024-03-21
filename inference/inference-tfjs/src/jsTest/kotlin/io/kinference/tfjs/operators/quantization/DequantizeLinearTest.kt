package io.kinference.tfjs.operators.quantization

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class DequantizeLinearTest {
    private fun getTargetPath(dirName: String) = "dequantize_linear/$dirName/"

    @Test
    fun test_linear_dequantization_defaults()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear"))
    }

    @Test
    fun test_linear_dequantization_per_tensor()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_tensor"))
    }

    @Test
    fun test_linear_dequantization_per_axis_1()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_axis_1"))
    }

    @Test
    fun test_linear_dequantization_per_axis()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_axis"))
    }
}
