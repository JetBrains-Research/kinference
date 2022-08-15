package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class ExpandTest {
    private fun getTargetPath(dirName: String) = "expand/$dirName/"

    @Test
    fun test_expand_dim_unchanged() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_unchanged"))
    }

    @Test
    fun test_expand_dim_changed() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_expand_dim_changed"))
    }
}
