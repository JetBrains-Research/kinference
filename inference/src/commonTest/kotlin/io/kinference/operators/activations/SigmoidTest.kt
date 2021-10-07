package io.kinference.operators.activations

import io.kinference.runners.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SigmoidTest {
    private fun getTargetPath(dirName: String) = "/sigmoid/$dirName/"

    @Test
    fun test_sigmoid_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sigmoid_example"))
    }

    @Test
    fun test_sigmoid() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sigmoid"))
    }
}
