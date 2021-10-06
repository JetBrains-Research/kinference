package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun test_matmul_2D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }
}
