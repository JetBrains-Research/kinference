package io.kinference.models.gpt

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test


class GPTGrazieTest {
    @Test
    fun heavy_test_gpt_grazie_model() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.00)
    }

    @Test
    fun benchmark_test_gpt_grazie_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("gpt2:grazie:distilled:quantized:v6")
    }
}
