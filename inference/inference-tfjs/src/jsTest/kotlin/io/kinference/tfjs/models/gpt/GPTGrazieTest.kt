package io.kinference.tfjs.models.gpt

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GPTGrazieTest {
    @Test
    fun heavy_test_gpt_grazie_model() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.0)
    }

    @Test
    fun benchmark_test_gpt_grazie_performance() = TestRunner.runTest {
        TFJSPerformanceRunner.runFromS3("gpt2:grazie:distilled:quantized:v6")
    }
}
