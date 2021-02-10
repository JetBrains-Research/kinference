package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.*

class GPTFLCCTest {
    @Disabled
    @Test
    @Tag("heavy")
    fun `test gpt py model`() = TestRunner.runTest {
        AccuracyRunner.runFromS3("gpt2:flcc-py-completion:quantized:v2", delta = 10.0)
    }

    @Test
    @Tag("heavy")
    fun `test gpt py performance`() = TestRunner.runTest {
        PerformanceRunner.runFromS3("gpt2:flcc-py-completion:quantized:v2", count = 50)
    }
}
