package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SumTest {
    private fun getTargetPath(dirName: String) = "sum/$dirName/"

    @Test
    fun test_sum_example()  = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sum_example"))
    }

    @Test
    fun test_sum_one_input()  = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sum_one_input"))
    }

    @Test
    fun test_sum_two_inputs()  = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sum_two_inputs"))
    }
}
