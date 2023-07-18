package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SinTest {
    private fun getTargetPath(dirName: String) = "sin/$dirName/"

    @Test
    fun test_sin() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sin"))
    }

    @Test
    fun test_sin_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sin_example"))
    }
}
