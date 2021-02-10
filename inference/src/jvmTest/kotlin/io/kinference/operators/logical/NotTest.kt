package io.kinference.operators.logical

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class NotTest {
    private fun getTargetPath(dirName: String) = "/not/$dirName/"

    @Test
    fun `test not for 2d tensor`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_not_2d"))
    }

    @Test
    fun `test not for 3d tensor`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_not_3d"))
    }

    @Test
    fun `test not for 4d tensor`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_not_4d"))
    }
}
