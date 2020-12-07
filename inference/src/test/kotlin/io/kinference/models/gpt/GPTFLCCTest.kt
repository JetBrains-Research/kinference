package io.kinference.models.gpt

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.*

class GPTFLCCTest {
    @Test
    @Tag("heavy")
    fun `test gpt py model`() {
        TestRunner.runFromS3("/gpt2/flcc-py-completion/quantized/v2/", "tests/gpt2/flcc-py-completion/quantized/v2", delta = 2.0)
    }

    @Test
    @Tag("heavy")
    fun `test gpt py performance`() {
        PerformanceRunner.runFromS3("/gpt2/flcc-py-completion/quantized/v2/", "tests/gpt2/flcc-py-completion/quantized/v2", count = 50)
    }
}
