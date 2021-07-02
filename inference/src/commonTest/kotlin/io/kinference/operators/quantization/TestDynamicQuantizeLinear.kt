package io.kinference.operators.quantization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

class TestDynamicQuantizeLinear {
    private fun getTargetPath(dirName: String) = "/dynamic_quantize_linear/$dirName/"

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_dynamic_quantize_linear_default()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_dynamic_quantize_linear_max_adjusted()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_max_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_dynamic_quantize_linear_min_adjusted()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_min_adjusted"), delta = AccuracyRunner.QUANT_DELTA)
    }
}
