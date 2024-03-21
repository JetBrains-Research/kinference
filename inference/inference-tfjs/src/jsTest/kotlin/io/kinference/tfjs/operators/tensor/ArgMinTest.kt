package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ArgMinTest {
    private fun getTargetPath(dirName: String) = "argmin/$dirName/"

    @Test
    fun test_argmin_default_axis_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_example"))
    }

    @Test
    fun test_argmin_default_axis_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_example_select_last_index"))
    }

    @Test
    fun test_argmin_default_axis_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_random"))
    }

    @Test
    fun test_argmin_default_axis_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_random_select_last_index"))
    }

    @Test
    fun test_argmin_keepdims_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_example"))
    }

    @Test
    fun test_argmin_keepdims_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_keepdims_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_random"))
    }

    @Test
    fun test_argmin_keepdims_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_example"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_random"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmin_no_keepdims_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_example"))
    }

    @Test
    fun test_argmin_no_keepdims_example_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_no_keepdims_random() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_random"))
    }

    @Test
    fun test_argmin_no_keepdims_random_select_last_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_random_select_last_index"))
    }
}
