package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ExpTest {
    private fun getTargetPath(dirName: String) = "exp/$dirName/"

    @Test
    fun test_exp() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_exp"))
    }

    @Test
    fun test_exp_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_exp_example"))
    }
}
