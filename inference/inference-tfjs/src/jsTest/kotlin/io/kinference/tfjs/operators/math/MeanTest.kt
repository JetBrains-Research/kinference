package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MeanTest {
    private fun getTargetPath(dirName: String) = "mean/$dirName/"

    @Test
    fun test_mean_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mean_example"))
    }

    @Test
    fun test_mean_one_input()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mean_one_input"))
    }

    @Test
    fun test_mean_two_inputs()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mean_two_inputs"))
    }
}
