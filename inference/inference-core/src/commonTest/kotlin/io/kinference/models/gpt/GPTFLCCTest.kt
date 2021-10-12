package io.kinference.models.gpt

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class GPTFLCCTest {
    //    @Test
    fun heavy_test_gpt_py_model() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("gpt2:flcc-py-completion:quantized:v2", delta = 10.0)
    }

    @Test
    fun benchmark_test_gpt_py_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("gpt2:flcc-py-completion:quantized:v2", count = 50)
    }
}
