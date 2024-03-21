package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "matmul/$dirName/"

    @Test
    fun test_matmul_2D()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3D()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4D()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }
}
