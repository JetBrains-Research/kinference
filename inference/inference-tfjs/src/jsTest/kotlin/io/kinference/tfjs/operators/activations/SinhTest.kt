package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SinhTest {
    private fun getTargetPath(dirName: String) = "sinh/$dirName/"

    @Test
    fun test_sinh() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sinh"))
    }

    @Test
    fun test_sinh_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sinh_example"))
    }
}
