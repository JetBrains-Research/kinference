package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class NotTest {
    private fun getTargetPath(dirName: String) = "not/$dirName/"

    @Test
    fun test_not_for_2d_tensor() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_not_2d"))
    }

    @Test
    fun test_not_for_3d_tensor() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_not_3d"))
    }

    @Test
    fun test_not_for_4d_tensor() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_not_4d"))
    }
}
