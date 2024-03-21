package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ExpTest {
    private fun getTargetPath(dirName: String) = "exp/$dirName/"

    @Test
    fun test_exp() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_exp"))
    }

    @Test
    fun test_exp_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_exp_example"))
    }
}
