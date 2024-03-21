package io.kinference.models.gpt

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GPTGrazieTest {
    @Test
    fun heavy_test_gpt_grazie_model() = runTest {
        val disabledTests = when (PlatformUtils.platform) {
            Platform.JVM -> listOf()
            Platform.JS -> listOf(
                "test_data_set_100_0",
                "test_data_set_100_1",
                "test_data_set_200_0",
                "test_data_set_200_1",
                "test_data_set_256_0",
                "test_data_set_256_1",
                "test_data_set_50_0",
            )
            else -> error(platformNotSupportedMessage)
        }
        KIAccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.00, disableTests = disabledTests)
    }

    @Test
    fun benchmark_test_gpt_grazie_performance() = runTest {
        KIPerformanceRunner.runFromS3("gpt2:grazie:distilled:quantized:v6")
    }
}
