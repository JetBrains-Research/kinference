package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MeanTest {
    private fun getTargetPath(dirName: String) = "mean/$dirName/"

    @Test
    fun test_mean_example()  = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_mean_example"))
    }

    @Test
    fun test_mean_one_input()  = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_mean_one_input"))
    }

    @Test
    fun test_mean_two_inputs()  = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_mean_two_inputs"))
    }
}
