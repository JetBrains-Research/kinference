package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ExpandTest {
    private fun getTargetPath(dirName: String) = "expand/$dirName/"

    @Test
    fun test_expand_dim_unchanged() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_unchanged"))
    }

    @Test
    fun test_expand_dim_changed() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_changed"))
    }
}
