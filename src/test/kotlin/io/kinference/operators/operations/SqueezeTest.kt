package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class SqueezeTest {
    private fun getTargetPath(dirName: String) = "/squeeze/$dirName/"

    @Test
    fun `test squeeze`() {
        Utils.tensorTestRunner(getTargetPath("test_squeeze"))
    }

    @Test
    fun `test squeeze with negative axes`() {
        Utils.tensorTestRunner(getTargetPath("test_squeeze_negative_axes"))
    }
}
