package io.kinference.misc.gpt

import io.kinference.Utils
import io.kinference.misc.ModelTestUtils
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class GPTTest {
    private fun getTargetPath(dirName: String) = "/gpt/$dirName/"

    @Test
    @Tag("heavy")
    fun `test GPT model`() {
        Utils.tensorTestRunner(getTargetPath("test_dummy_input"), 1.07)
    }

    @Test
    @Tag("heavy")
    fun `test GPT performance`() {
        ModelTestUtils.testModelPerformance("/gpt/test_dummy_input/")
    }
}
