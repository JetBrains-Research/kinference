package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LeakyReluTest {
    private fun getTargetPath(dirName: String) = "leakyrelu/$dirName/"

    @Test
    fun test_leaky_relu() =
        runTest {
            KIAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu"))
        }

    @Test
    fun test_leaky_relu_default() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_default"))
    }

    @Test
    fun test_leaky_relu_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_example"))
    }
}
