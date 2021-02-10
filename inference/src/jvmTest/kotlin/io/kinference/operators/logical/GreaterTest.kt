package io.kinference.operators.logical

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class GreaterTest {
    private fun getTargetPath(dirName: String) = "/greater/$dirName/"

    @Test
    fun `test greater`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_greater"))
    }

    @Test
    fun `test greater with broadcast`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_greater_bcast"))
    }
}
