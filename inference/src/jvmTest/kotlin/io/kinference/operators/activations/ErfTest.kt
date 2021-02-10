package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class ErfTest {
    private fun getTargetPath(dirName: String) = "/erf/$dirName/"

    @Test
    fun `test erf`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_erf"))
    }
}
