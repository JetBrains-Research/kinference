package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SigmoidTest {
    private fun getTargetPath(dirName: String) = "sigmoid/$dirName/"

    @Test
    fun test_sigmoid_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sigmoid_example"))
    }

    @Test
    fun test_sigmoid() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sigmoid"))
    }
}
