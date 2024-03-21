package io.kinference.operators.operations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ArgMinTest {
    private fun getTargetPath(dirName: String) = "argmin/$dirName/"

    @Test
    fun test_argmin_default_axis_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_example"))
    }

    @Test
    fun test_argmin_default_axis_example_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_example_select_last_index"))
    }

    @Test
    fun test_argmin_default_axis_random() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_random"))
    }

    @Test
    fun test_argmin_default_axis_random_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_random_select_last_index"))
    }

    @Test
    fun test_argmin_keepdims_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_example"))
    }

    @Test
    fun test_argmin_keepdims_example_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_keepdims_random() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_random"))
    }

    @Test
    fun test_argmin_keepdims_random_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_example"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_example_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_random() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_random"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_random_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmin_no_keepdims_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_example"))
    }

    @Test
    fun test_argmin_no_keepdims_example_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_no_keepdims_random() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_random"))
    }

    @Test
    fun test_argmin_no_keepdims_random_select_last_index() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_random_select_last_index"))
    }
}
