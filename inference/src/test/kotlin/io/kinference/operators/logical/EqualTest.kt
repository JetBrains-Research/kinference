package io.kinference.operators.logical

import io.kinference.Utils
import org.junit.jupiter.api.Test

class EqualTest {
    private fun getTargetPath(dirName: String) = "/equal/$dirName/"

    @Test
    fun `test equal`() {
        Utils.tensorTestRunner(getTargetPath("test_equal"))
    }

    @Test
    fun `test equal with broadcast`() {
        Utils.tensorTestRunner(getTargetPath("test_equal_bcast"))
    }
}
