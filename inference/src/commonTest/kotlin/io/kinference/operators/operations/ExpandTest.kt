package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class ExpandTest {
    private fun getTargetPath(dirName: String) = "/expand/$dirName/"

    @Test
    fun test_expand_dim_unchanged() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_expand_dim_unchanged"))
    }

    @Test
    fun test_expand_dim_changed() = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_expand_dim_changed"))
    }
}
