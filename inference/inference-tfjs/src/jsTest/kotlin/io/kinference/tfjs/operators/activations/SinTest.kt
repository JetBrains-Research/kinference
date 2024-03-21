package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SinTest {
    private fun getTargetPath(dirName: String) = "sin/$dirName/"

    @Test
    fun test_sin() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sin"))
    }

    @Test
    fun test_sin_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sin_example"))
    }
}
