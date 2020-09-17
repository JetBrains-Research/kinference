package io.kinference.operators.math

import io.kinference.Utils
import org.junit.jupiter.api.Test

class MatMulIntegerTest {
    private fun getTargetPath(dirName: String) = "/matmul_integer/$dirName/"

    @Test
    fun `test matmul integer`() {
        Utils.tensorTestRunner(getTargetPath("test_matmulinteger"))
    }

    @Test
    fun `test matmul integer op`() {
        Utils.tensorTestRunner(getTargetPath("test_matmulinteger_op"))
    }
}
