package org.jetbrains.research.kotlin.inference.operators.math

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun test_matmul_2d() {
        Utils.tensorTestRunner(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3d() {
        Utils.tensorTestRunner(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4d() {
        Utils.tensorTestRunner(getTargetPath("test_matmul_4d"))
    }
}
