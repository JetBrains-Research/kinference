package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class GPTRTest {
    @Test
    @Tag("heavy")
    fun `test GPT model`() {
        TestRunner.runFromS3("/gpt/r/standard/v1/", "tests/gpt2/r-completion/standard/v1", 0.2)
    }

    @Test
    @Tag("heavy")
    fun `test GPT performance`() {
        PerformanceRunner.runFromS3("/gpt/r/standard/v1/", "tests/gpt2/r-completion/standard/v1")
    }


    @Test
    @Tag("heavy")
    fun `test GPT quantized model`() {
        TestRunner.runFromS3("/gpt/r/quantized/v1/", "tests/gpt2/r-completion/quantized/v1", 0.2)
    }
    @Test
    @Tag("heavy")
    fun `test GPT quantized performance`() {
        PerformanceRunner.runFromS3("/gpt/r/quantized/v1/", "tests/gpt2/r-completion/quantized/v1")
    }
}
