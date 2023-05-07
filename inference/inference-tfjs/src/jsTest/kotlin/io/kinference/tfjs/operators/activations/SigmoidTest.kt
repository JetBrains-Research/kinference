package io.kinference.tfjs.operators.activations

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SigmoidTest {
    private fun getTargetPath(dirName: String) = "sigmoid/$dirName/"

    @Test
    fun test_sigmoid_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sigmoid_example"))
    }

    @Test
    fun test_sigmoid() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sigmoid"))
    }
}
