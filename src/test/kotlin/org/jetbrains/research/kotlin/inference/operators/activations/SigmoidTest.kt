package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.Utils
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
