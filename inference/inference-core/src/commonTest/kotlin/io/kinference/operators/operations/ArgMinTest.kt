package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ArgMinTest {
    private fun getTargetPath(dirName: String) = "argmin/$dirName/"

    @Test
    fun test_argmin_default_axis_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_example"))
    }

    @Test
    fun test_argmin_default_axis_example_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_example_select_last_index"))
    }

    @Test
    fun test_argmin_default_axis_random() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_random"))
    }

    @Test
    fun test_argmin_default_axis_random_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_default_axis_random_select_last_index"))
    }

    @Test
    fun test_argmin_keepdims_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_example"))
    }

    @Test
    fun test_argmin_keepdims_example_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_keepdims_random() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_random"))
    }

    @Test
    fun test_argmin_keepdims_random_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_example"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_example_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_random() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_random"))
    }

    @Test
    fun test_argmin_negative_axis_keepdims_random_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_negative_axis_keepdims_random_select_last_index"))
    }

    @Test
    fun test_argmin_no_keepdims_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_example"))
    }

    @Test
    fun test_argmin_no_keepdims_example_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_example_select_last_index"))
    }

    @Test
    fun test_argmin_no_keepdims_random() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_random"))
    }

    @Test
    fun test_argmin_no_keepdims_random_select_last_index() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_argmin_no_keepdims_random_select_last_index"))
    }
}
