package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class NotTest {
    private fun getTargetPath(dirName: String) = "/not/$dirName/"

    @Test
    fun `test not for 2d tensor`() {
        Utils.tensorTestRunner(getTargetPath("test_not_2d"))
    }

    @Test
    fun `test not for 3d tensor`() {
        Utils.tensorTestRunner(getTargetPath("test_not_3d"))
    }

    @Test
    fun `test not for 4d tensor`() {
        Utils.tensorTestRunner(getTargetPath("test_not_4d"))
    }
}
