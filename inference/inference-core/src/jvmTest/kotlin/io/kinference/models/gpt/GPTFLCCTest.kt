package io.kinference.models.gpt

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test


class GPTFLCCTest {
    @Test
    fun heavy_test_gpt_py_model_v3() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("gpt2:flcc-py-completion:standard:v3")
    }

    @Test
    fun heavy_test_gpt_py_model_quantized_v3() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("gpt2:flcc-py-completion:quantized:v3", delta = 1.5)
    }

    @Test
    fun benchmark_test_gpt_py_model_v3() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("gpt2:flcc-py-completion:standard:v3", count = 20)
    }

    @Test
    fun benchmark_test_gpt_py_model_quantized_v3() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("gpt2:flcc-py-completion:quantized:v3", count = 20)
    }

}
