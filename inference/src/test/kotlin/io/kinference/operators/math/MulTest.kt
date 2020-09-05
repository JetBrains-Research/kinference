package io.kinference.operators.math

import io.kinference.Utils
import org.junit.jupiter.api.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "/mul/$dirName/"

    @Test
    fun `test mul`() {
        Utils.tensorTestRunner(getTargetPath("test_mul"))
    }

    @Test
    fun `test mul with broadcast`() {
        Utils.tensorTestRunner(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun `test mul defaults`() {
        Utils.tensorTestRunner(getTargetPath("test_mul_example"))
    }
}
