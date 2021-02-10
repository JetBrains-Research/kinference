package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class GPTRTest {
    @Test
    @Tag("heavy")
    fun `test GPT model`()  = TestRunner.runTest {
        AccuracyRunner.runFromS3("gpt2:r-completion:standard:v1")
    }

    @Test
    @Tag("heavy")
    fun `test GPT performance`()  = TestRunner.runTest {
        PerformanceRunner.runFromS3("gpt2:r-completion:standard:v1")
    }


    @Test
    @Tag("heavy")
    fun `test GPT quantized model`()  = TestRunner.runTest {
        AccuracyRunner.runFromS3("gpt2:r-completion:quantized:v1", delta = 2.4)
    }
    @Test
    @Tag("heavy")
    fun `test GPT quantized performance`()  = TestRunner.runTest {
        PerformanceRunner.runFromS3("gpt2:r-completion:quantized:v1")
    }
}
