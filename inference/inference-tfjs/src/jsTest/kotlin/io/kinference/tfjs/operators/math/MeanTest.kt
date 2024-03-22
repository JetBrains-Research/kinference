package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MeanTest {
    private fun getTargetPath(dirName: String) = "mean/$dirName/"

    @Test
    fun test_mean_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mean_example"))
    }

    @Test
    fun test_mean_one_input()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mean_one_input"))
    }

    @Test
    fun test_mean_two_inputs()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mean_two_inputs"))
    }
}
