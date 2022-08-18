package io.kinference.tfjs.operators.logical

import io.kinference.utils.TestRunner
import kotlin.test.Test
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner

class NotTest {
    private fun getTargetPath(dirName: String) = "not/$dirName/"

    @Test
    fun test_not_for_2d_tensor() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_not_2d"))
    }

    @Test
    fun test_not_for_3d_tensor() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_not_3d"))
    }

    @Test
    fun test_not_for_4d_tensor() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_not_4d"))
    }
}
