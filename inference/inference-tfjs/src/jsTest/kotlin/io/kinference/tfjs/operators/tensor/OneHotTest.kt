package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class OneHotTest {
    private fun getTargetPath(dirName: String) = "onehot/$dirName/"

    @Test
    fun test_onehot_with_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_onehot_with_axis"))
    }

    @Test
    fun test_onehot_without_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_onehot_without_axis"))
    }

    @Test
    fun test_onehot_with_negative_axis() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_onehot_with_negative_axis"))
    }

    @Test
    fun test_onehot_negative_indices() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_onehot_negative_indices"))
    }
}