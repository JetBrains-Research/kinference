package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GPTFLCCTest {
    @Test
    fun heavy_test_gpt_py_model() = TestRunner.runTest {
        AccuracyRunner.runFromS3("gpt2:flcc-py-completion:quantized:v2", delta = 10.0)
    }

    @Test
    fun heavy_test_gpt_py_performance() = TestRunner.runTest {
        PerformanceRunner.runFromS3("gpt2:flcc-py-completion:quantized:v2", count = 50)
    }
}
