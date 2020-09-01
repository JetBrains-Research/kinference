package io.kinference.operators.activations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class ErfTest {
    private fun getTargetPath(dirName: String) = "/erf/$dirName/"

    @Test
    fun `test erf`() {
        Utils.tensorTestRunner(getTargetPath("test_erf"))
    }
}
