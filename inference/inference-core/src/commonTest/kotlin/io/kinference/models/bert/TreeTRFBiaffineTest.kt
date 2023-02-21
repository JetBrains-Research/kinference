package io.kinference.models.bert

import io.kinference.KITestEngine
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TreeTRFBiaffineTest {
    @Test
    fun heavy_test_jvm_tree_trf_biaffine_quantized() = TestRunner.runTest(Platform.JVM) {
        KITestEngine.KIAccuracyRunner.runFromS3("bert:en_tree:quantized", delta = 3.8)
    }

//    @Test
    fun heavy_test_js_tree_trf_biaffine_quantized() = TestRunner.runTest(Platform.JS) {
        KITestEngine.KIAccuracyRunner.runFromS3("bert:en_tree:quantized", delta = 3.8, disableTests = listOf(
            "test_data_64_4",
            "test_data_32_16",
            "test_data_32_4",
            "test_data_32_3",
        ))
    }

    @Test
    fun benchmark_test_tree_trf_biaffine_quantized() = TestRunner.runTest {
        KITestEngine.KIPerformanceRunner.runFromS3("bert:en_tree:quantized")
    }
}
