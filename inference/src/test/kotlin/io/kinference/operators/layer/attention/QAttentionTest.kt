package io.kinference.operators.layer.attention

import io.kinference.Utils
import org.junit.jupiter.api.Test

class QAttentionTest {
    private fun getTargetPath(dirName: String) = "/qattention/$dirName/"

    @Test
    fun `test quantized attention defaults`() {
        Utils.tensorTestRunner(getTargetPath("test_qattention_defaults"))
    }

    @Test
    fun `test quantized attention`() {
        Utils.tensorTestRunner(getTargetPath("test_qattention_op"))
    }
}
