package io.kinference.operators.quantization

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.runners.AccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TestDynamicQuantizeLinear {
    private fun getTargetPath(dirName: String) = "dynamic_quantize_linear/$dirName/"

    @Test
    fun test_dynamic_quantize_linear_default() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_dynamic_quantize_linear_max_adjusted() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_max_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun test_dynamic_quantize_linear_min_adjusted() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_min_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
