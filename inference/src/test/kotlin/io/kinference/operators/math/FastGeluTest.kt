package io.kinference.operators.math

import io.kinference.Utils
import org.junit.jupiter.api.Test

class FastGeluTest {
    private fun getTargetPath(dirName: String) = "/fastgelu/$dirName/"

    @Test
    fun `test fast GELU with bias`() {
        Utils.tensorTestRunner(getTargetPath("test_fastgelu_with_bias"))
    }

    @Test
    fun `test fast GELU without bias`() {
        Utils.tensorTestRunner(getTargetPath("test_fastgelu_without_bias"))
    }
}
