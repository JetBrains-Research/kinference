package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SumTest {
    private fun getTargetPath(dirName: String) = "sum/$dirName/"

    @Test
    fun test_sum_example()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sum_example"))
    }

    @Test
    fun test_sum_one_input()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sum_one_input"))
    }

    @Test
    fun test_sum_two_inputs()  = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sum_two_inputs"))
    }
}
