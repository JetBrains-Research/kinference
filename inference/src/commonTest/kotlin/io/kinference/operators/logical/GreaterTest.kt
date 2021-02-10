package io.kinference.operators.logical

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GreaterTest {
    private fun getTargetPath(dirName: String) = "/greater/$dirName/"

    @Test
    fun test_greater()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_greater"))
    }

    @Test
    fun test_greater_with_broadcast()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_greater_bcast"))
    }
}
