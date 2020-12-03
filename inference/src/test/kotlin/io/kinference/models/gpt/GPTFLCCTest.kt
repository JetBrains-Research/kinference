package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.*

class GPTFLCCTest {
    @Disabled
    @Test
    @Tag("heavy")
    fun `test gpt py model`() {
        TestRunner.runFromS3("/gpt2/flcc-py-completion/quantized/v1/", "tests/gpt2/flcc-py-completion/quantized/v1", delta = 0.7)
    }

    @Disabled
    @Test
    @Tag("heavy")
    fun `test gpt py performance`() {
        PerformanceRunner.runFromS3("/gpt2/flcc-py-completion/quantized/v1/", "tests/gpt2/flcc-py-completion/quantized/v1")
    }
}
