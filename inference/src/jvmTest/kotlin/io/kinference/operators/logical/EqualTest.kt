package io.kinference.operators.logical

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class EqualTest {
    private fun getTargetPath(dirName: String) = "/equal/$dirName/"

    @Test
    fun `test equal`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_equal"))
    }

    @Test
    fun `test equal with broadcast`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_equal_bcast"))
    }
}
