package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class ExpandTest {
    private fun getTargetPath(dirName: String) = "expand/$dirName/"

    @Test
    fun test_expand_dim_unchanged() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_unchanged"))
    }

    @Test
    fun test_expand_dim_changed() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_changed"))
    }
}
