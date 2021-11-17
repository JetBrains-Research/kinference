package io.kinference.models.bert

import io.kinference.KITestEngine
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TreeTRFBiaffineTest {
    @Test
    fun heavy_test_tree_trf_biaffine_quantized() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromS3("bert:en_tree:quantized", delta = 3.8)
    }

    @Test
    fun benchmark_test_tree_trf_biaffine_quantized() = TestRunner.runTest {
        KITestEngine.KIPerformanceRunner.runFromS3("bert:en_tree:quantized")
    }
}
