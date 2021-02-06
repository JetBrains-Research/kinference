package io.kinference.operators.math

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun `test matmul 2D`() {
        TestRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun `test matmul 3D`() {
        TestRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun `test matmul 4D`() {
        TestRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }
}
