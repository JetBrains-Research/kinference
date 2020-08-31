package io.kinference.operators.math

import io.kinference.Utils
import org.junit.jupiter.api.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun `test matmul 2D`() {
        Utils.tensorTestRunner(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun `test matmul 3D`() {
        Utils.tensorTestRunner(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun `test matmul 4D`() {
        Utils.tensorTestRunner(getTargetPath("test_matmul_4d"))
    }
}
