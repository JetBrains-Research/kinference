package io.kinference.operators.activations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class IdentityTest {
    private fun getTargetPath(dirName: String) = "/identity/$dirName/"

    @Test
    fun `test identity`() {
        TestRunner.runFromResources(getTargetPath("test_identity"))
    }
}
