package io.kinference.operators.logical

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class EqualTest {
    private fun getTargetPath(dirName: String) = "/equal/$dirName/"

    @Test
    fun test_equal()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_equal"))
    }

    @Test
    fun test_equal_with_broadcast()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_equal_bcast"))
    }
}
