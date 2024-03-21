package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class ExpandTest {
    private fun getTargetPath(dirName: String) = "expand/$dirName/"

    @Test
    fun test_expand_dim_unchanged() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_unchanged"))
    }

    @Test
    fun test_expand_dim_changed() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_changed"))
    }
}
