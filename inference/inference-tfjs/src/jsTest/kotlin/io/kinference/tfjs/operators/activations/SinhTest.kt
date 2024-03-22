package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SinhTest {
    private fun getTargetPath(dirName: String) = "sinh/$dirName/"

    @Test
    fun test_sinh() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sinh"))
    }

    @Test
    fun test_sinh_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sinh_example"))
    }
}
