package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class GPTGrazieTest {
    @Test
    @Tag("heavy")
    fun `test GPT model`() = TestRunner.runTest {
        AccuracyRunner.runFromS3("gpt2:grazie:distilled:quantized:v6", delta = 5.00)
    }

    @Test
    @Tag("heavy")
    fun `test GPT performance`() = TestRunner.runTest {
        PerformanceRunner.runFromS3("gpt2:grazie:distilled:quantized:v6")
    }
}
