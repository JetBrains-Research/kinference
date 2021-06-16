package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class GPTGrazieTest {
    @Test
    fun heavy_test_gpt_grazie_model() = TestRunner.runTest {
        AccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.00)
    }

    @Test
    fun benchmark_test_gpt_grazie_performance() = TestRunner.runTest {
        PerformanceRunner.runFromS3("gpt2:grazie:distilled:quantized:v6")
    }
}
