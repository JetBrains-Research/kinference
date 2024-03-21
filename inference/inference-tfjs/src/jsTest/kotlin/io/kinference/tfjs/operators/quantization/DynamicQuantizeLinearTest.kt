package io.kinference.tfjs.operators.quantization

import io.kinference.runners.AccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class DynamicQuantizeLinearTest {
    private fun getTargetPath(dirName: String) = "dynamic_quantize_linear/$dirName/"

    @Test
    fun test_dynamic_quantize_linear_default()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_dynamic_quantize_linear_max_adjusted()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_max_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_dynamic_quantize_linear_min_adjusted()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_min_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
