package io.kinference.ort.models.gpt

import io.kinference.ort.ORTTestEngine
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Disabled
import kotlin.test.Test

@Disabled
class GPTRTest {
    @Test
    fun gpu_test_gpt_model() = TestRunner.runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromS3("gpt2:r-completion:standard:v1")
    }

    @Test
    fun benchmark_test_gpt_performance() = TestRunner.runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromS3("gpt2:r-completion:standard:v1")
    }


    @Test
    fun gpu_test_gpt_quantized_model() = TestRunner.runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromS3("gpt2:r-completion:quantized:v1", delta = AccuracyRunner.QUANT_DELTA)
    }

    @Test
    fun benchmark_test_gpt_quantized_performance() = TestRunner.runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromS3("gpt2:r-completion:quantized:v1")
    }
}
