package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class GatherTest {
    private fun getTargetPath(dirName: String) = "/gather/$dirName/"

    @Test
    fun `test gather 0`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_gather_0"))
    }

    @Test
    fun `test gather 1`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_gather_1"))
    }

    @Test
    fun `test gather with negative indices`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_gather_negative_indices"))
    }
}

