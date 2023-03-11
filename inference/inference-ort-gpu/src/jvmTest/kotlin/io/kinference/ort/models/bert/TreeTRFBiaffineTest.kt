package io.kinference.ort.models.bert

import io.kinference.ort.ORTTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test


class TreeTRFBiaffineTest {
    @Test
    fun gpu_test_tree_trf_biaffine_quantized() = TestRunner.runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromS3("bert:en_tree:quantized", delta = 4.9)
    }

    @Test
    fun benchmark_test_tree_trf_biaffine_quantized() = TestRunner.runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromS3("bert:en_tree:quantized")
    }
}
