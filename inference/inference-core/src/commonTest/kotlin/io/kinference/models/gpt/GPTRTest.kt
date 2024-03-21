package io.kinference.models.gpt

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class GPTRTest {
    @Test
    fun heavy_test_gpt_model() = runTest {
        val disabledTests = when (PlatformUtils.platform) {
            Platform.JVM -> listOf()
            Platform.JS -> listOf(
                "test_data_set_0_200",
                "test_data_set_0_50",
                "test_data_set_0_512",
                "test_data_set_1_200",
                "test_data_set_1_512"
            )
            else -> error(platformNotSupportedMessage)
        }
        KIAccuracyRunner.runFromS3("gpt2:r-completion:standard:v1", disableTests = disabledTests)
    }

    @Test
    fun benchmark_test_gpt_performance() = runTest {
        KIPerformanceRunner.runFromS3("gpt2:r-completion:standard:v1")
    }


    @Test
    fun heavy_test_gpt_quantized_model() = runTest {
        when (PlatformUtils.platform) {
            Platform.JVM -> KIAccuracyRunner.runFromS3("gpt2:r-completion:quantized:v1", delta = AccuracyRunner.QUANT_DELTA)
            else -> { }
        }
    }

    @Test
    fun benchmark_test_gpt_quantized_performance() = runTest {
        KIPerformanceRunner.runFromS3("gpt2:r-completion:quantized:v1")
    }
}
