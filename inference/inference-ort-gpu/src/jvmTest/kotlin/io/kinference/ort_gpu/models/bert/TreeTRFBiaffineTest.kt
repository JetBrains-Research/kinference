package io.kinference.ort_gpu.models.bert

import io.kinference.ort_gpu.ORTGPUTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TreeTRFBiaffineTest {
    @Test
    fun heavy_test_tree_trf_biaffine_quantized() = TestRunner.runTest {
        ORTGPUTestEngine.ORTGPUAccuracyRunner.runFromS3("bert:en_tree:quantized", delta = 4.9)
    }

    @Test
    fun benchmark_test_tree_trf_biaffine_quantized() = TestRunner.runTest {
        ORTGPUTestEngine.ORTGPUPerformanceRunner.runFromS3("bert:en_tree:quantized")
    }
}
