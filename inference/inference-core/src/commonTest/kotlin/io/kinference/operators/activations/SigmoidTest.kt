package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SigmoidTest {
    private fun getTargetPath(dirName: String) = "sigmoid/$dirName/"

    @Test
    fun test_sigmoid_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sigmoid_example"))
    }

    @Test
    fun test_sigmoid() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_sigmoid"))
    }
}
