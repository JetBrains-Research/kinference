package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "matmul/$dirName/"

    @Test
    fun test_matmul_2D() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3D() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4D() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }
}
