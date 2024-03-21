package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ArgMaxTest {
    private fun getTargetPath(dirName: String) = "argmax/$dirName/"

    @Test
    fun test_argmax_default_axis_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_default_axis_example"))
    }

    @Test
    fun test_argmax_default_axis_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_default_axis_example_select_last_index"))
    }

    @Test
    fun test_argmax_default_axis_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_default_axis_random"))
    }

    @Test
    fun test_argmax_default_axis_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_default_axis_random_select_last_index"))
    }

    @Test
    fun test_argmax_keepdims_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_keepdims_example"))
    }

    @Test
    fun test_argmax_keepdims_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmax_keepdims_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_keepdims_random"))
    }

    @Test
    fun test_argmax_keepdims_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmax_negative_axis_keepdims_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_negative_axis_keepdims_example"))
    }

    @Test
    fun test_argmax_negative_axis_keepdims_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_negative_axis_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmax_negative_axis_keepdims_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_negative_axis_keepdims_random"))
    }

    @Test
    fun test_argmax_negative_axis_keepdims_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_negative_axis_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmax_no_keepdims_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_no_keepdims_example"))
    }

    @Test
    fun test_argmax_no_keepdims_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_no_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmax_no_keepdims_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_no_keepdims_random"))
    }

    @Test
    fun test_argmax_no_keepdims_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmax_no_keepdims_random_select_last_index"))
    }
}
