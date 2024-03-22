package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MeanTest {
    private fun getTargetPath(dirName: String) = "mean/$dirName/"

    @Test
    fun test_mean_example()  = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_mean_example"))
    }

    @Test
    fun test_mean_one_input()  = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_mean_one_input"))
    }

    @Test
    fun test_mean_two_inputs()  = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_mean_two_inputs"))
    }
}
