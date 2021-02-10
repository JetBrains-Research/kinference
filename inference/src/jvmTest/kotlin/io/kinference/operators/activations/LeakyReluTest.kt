package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class LeakyReluTest {
    private fun getTargetPath(dirName: String) = "/leakyrelu/$dirName/"

    @Test
    fun `test LeakyRelu`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_leakyrelu"))
    }

    @Test
    fun `test LeakyRelu default`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_default"))
    }

    @Test
    fun `test LeakyRelu example`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_leakyrelu_example"))
    }
}
