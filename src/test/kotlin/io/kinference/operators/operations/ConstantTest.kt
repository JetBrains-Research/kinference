package io.kinference.operators.operations

import io.kinference.Utils
import org.junit.jupiter.api.Test

class ConstantTest {
    private fun getTargetPath(dirName: String) = "/constant/$dirName/"

    @Test
    fun `test constant`() {
        Utils.tensorTestRunner(getTargetPath("test_constant"))
    }

    @Test
    fun `test scalar constant`() {
        Utils.tensorTestRunner(getTargetPath("test_scalar_constant"))
    }
}
