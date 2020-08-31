package io.kinference.operators.activations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class SigmoidTest {
    private fun getTargetPath(dirName: String) = "/sigmoid/$dirName/"

    @Test
    fun `test sigmoid example`() {
        Utils.tensorTestRunner(getTargetPath("test_sigmoid_example"))
    }

    @Test
    fun `test sigmoid`() {
        Utils.tensorTestRunner(getTargetPath("test_sigmoid"))
    }
}
