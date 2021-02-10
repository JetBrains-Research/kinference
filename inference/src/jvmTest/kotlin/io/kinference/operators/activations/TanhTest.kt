package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class TanhTest {
    private fun getTargetPath(dirName: String) = "/tanh/$dirName/"

    @Test
    fun `test tanh example`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_tanh_example"))
    }

    @Test
    fun `test tanh`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_tanh"))
    }

    @Test
    fun `test tanh scalar`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_tanh_scalar"))
    }
}
