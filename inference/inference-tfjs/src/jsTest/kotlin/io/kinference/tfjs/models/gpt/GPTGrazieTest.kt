package io.kinference.tfjs.models.gpt

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GPTGrazieTest {
    @Test
    fun heavy_test_gpt_grazie_model() = runTest {
        TFJSAccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.0)
    }

    @Test
    fun benchmark_test_gpt_grazie_performance() = runTest {
        TFJSPerformanceRunner.runFromS3("gpt2:grazie:distilled:quantized:v6")
    }
}
