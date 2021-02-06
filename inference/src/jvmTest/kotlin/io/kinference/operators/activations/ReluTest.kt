package io.kinference.operators.activations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class ReluTest {
    private fun getTargetPath(dirName: String) = "/relu/$dirName/"

    @Test
    fun `test relu`() {
        TestRunner.runFromResources(getTargetPath("test_relu"))
    }
}
