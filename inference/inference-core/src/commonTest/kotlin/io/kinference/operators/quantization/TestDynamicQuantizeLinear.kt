package io.kinference.operators.quantization

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TestDynamicQuantizeLinear {
    private fun getTargetPath(dirName: String) = "dynamic_quantize_linear/$dirName/"

        @Test
    fun test_dynamic_quantize_linear_default() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear"), delta = AccuracyRunner.QUANT_DELTA)
    }

        @Test
    fun test_dynamic_quantize_linear_max_adjusted() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_max_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }

        @Test
    fun test_dynamic_quantize_linear_min_adjusted() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_min_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
