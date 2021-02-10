package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class SigmoidTest {
    private fun getTargetPath(dirName: String) = "/sigmoid/$dirName/"

    @Test
    fun `test sigmoid example`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_sigmoid_example"))
    }

    @Test
    fun `test sigmoid`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_sigmoid"))
    }
}
