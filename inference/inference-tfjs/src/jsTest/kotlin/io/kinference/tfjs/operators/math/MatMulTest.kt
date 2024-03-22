package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "matmul/$dirName/"

    @Test
    fun test_matmul_2D()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3D()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4D()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }
}
