package io.kinference.operators.activations

import io.kinference.runners.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LeakyReluTest {
    private fun getTargetPath(dirName: String) = "/leakyrelu/$dirName/"

    @Test
    fun test_leaky_relu() =
        TestRunner.runTest {
            KIAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu"))
        }

    @Test
    fun test_leaky_relu_default() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_default"))
    }

    @Test
    fun test_leaky_relu_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_example"))
    }
}
