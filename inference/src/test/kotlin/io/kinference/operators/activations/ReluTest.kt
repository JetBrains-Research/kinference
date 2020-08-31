package io.kinference.operators.activations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class ReluTest {
    private fun getTargetPath(dirName: String) = "/relu/$dirName/"

    @Test
    fun `test relu`() {
        Utils.tensorTestRunner(getTargetPath("test_relu"))
    }
}
