package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class ReluTest {
    private fun getTargetPath(dirName: String) = "/relu/$dirName/"

    @Test
    fun `test relu`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_relu"))
    }
}
