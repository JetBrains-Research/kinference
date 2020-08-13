package org.jetbrains.research.kotlin.inference.operators.math

import org.jetbrains.research.kotlin.inference.Utils
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
