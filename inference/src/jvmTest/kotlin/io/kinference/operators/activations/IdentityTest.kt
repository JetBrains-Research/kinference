package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class IdentityTest {
    private fun getTargetPath(dirName: String) = "/identity/$dirName/"

    @Test
    fun `test identity`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_identity"))
    }
}
