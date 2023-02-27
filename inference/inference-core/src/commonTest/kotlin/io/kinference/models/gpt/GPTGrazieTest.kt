package io.kinference.models.gpt

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class GPTGrazieTest {
    @Test
    fun heavy_test_jvm_gpt_grazie_model() = TestRunner.runTest(Platform.JVM) {
        KIAccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.00)
    }

    @Test
    fun heavy_test_js_gpt_grazie_model() = TestRunner.runTest(Platform.JS) {
        KIAccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.00, disableTests = listOf(
            "test_data_set_100_0",
            "test_data_set_100_1",
            "test_data_set_200_0",
            "test_data_set_200_1",
            "test_data_set_256_0",
            "test_data_set_256_1",
            "test_data_set_50_0",
        ))
    }

    @Test
    fun benchmark_test_gpt_grazie_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("gpt2:grazie:distilled:quantized:v6")
    }
}
