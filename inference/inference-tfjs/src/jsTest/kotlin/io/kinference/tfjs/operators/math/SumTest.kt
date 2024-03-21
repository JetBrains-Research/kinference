package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SumTest {
    private fun getTargetPath(dirName: String) = "sum/$dirName/"

    @Test
    fun test_sum_example()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sum_example"))
    }

    @Test
    fun test_sum_one_input()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sum_one_input"))
    }

    @Test
    fun test_sum_two_inputs()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sum_two_inputs"))
    }
}
