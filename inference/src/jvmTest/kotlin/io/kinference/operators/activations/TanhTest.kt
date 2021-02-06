package io.kinference.operators.activations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class TanhTest {
    private fun getTargetPath(dirName: String) = "/tanh/$dirName/"

    @Test
    fun `test tanh example`() {
        TestRunner.runFromResources(getTargetPath("test_tanh_example"))
    }

    @Test
    fun `test tanh`() {
        TestRunner.runFromResources(getTargetPath("test_tanh"))
    }

    @Test
    fun `test tanh scalar`() {
        TestRunner.runFromResources(getTargetPath("test_tanh_scalar"))
    }
}
