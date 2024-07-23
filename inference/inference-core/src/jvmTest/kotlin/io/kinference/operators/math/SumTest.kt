package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SumTest {
    private fun getTargetPath(dirName: String) = "sum/$dirName/"

    @Test
    fun test_sum_example()  = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sum_example"))
    }

    @Test
    fun test_sum_one_input()  = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sum_one_input"))
    }

    @Test
    fun test_sum_two_inputs()  = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sum_two_inputs"))
    }
}
