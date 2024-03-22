package io.kinference.models.bert

import io.kinference.KITestEngine
import io.kinference.utils.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration

class TreeTRFBiaffineTest {
    @Test
    fun heavy_test_tree_trf_biaffine_quantized() = runTest(timeout = Duration.INFINITE) {
        val disabledTests = when (PlatformUtils.platform) {
            Platform.JVM -> listOf()
            Platform.JS -> listOf(
                "test_data_64_4",
                "test_data_32_16",
                "test_data_32_4",
                "test_data_32_3",
            )
            else -> error(platformNotSupportedMessage)
        }
        KITestEngine.KIAccuracyRunner.runFromS3("bert:en_tree:quantized", delta = 3.8, disableTests = disabledTests)
    }

    @Test
    fun benchmark_test_tree_trf_biaffine_quantized() = runTest(timeout = Duration.INFINITE) {
        KITestEngine.KIPerformanceRunner.runFromS3("bert:en_tree:quantized")
    }
}
