package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun `test matmul 2D`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun `test matmul 3D`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun `test matmul 4D`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }
}
