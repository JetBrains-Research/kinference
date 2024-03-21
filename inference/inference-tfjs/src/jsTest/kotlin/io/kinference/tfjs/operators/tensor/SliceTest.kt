package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SliceTest {
    private fun getTargetPath(dirName: String) = "slice/$dirName/"

    @Test
    fun test_slice() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice"))
    }

    @Test
    fun test_slice_with_default_axes() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice_default_axes"))
    }

    @Test
    fun test_slice_with_default_steps() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice_default_steps"))
    }

    @Test
    fun test_slice_end_out_of_bounds() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice_end_out_of_bounds"))
    }

    @Test
    fun test_slice_with_negative_index() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice_neg"))
    }

    @Test
    fun test_slice_with_negative_steps() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice_neg_steps"))
    }

    @Test
    fun test_slice_with_negative_axes() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice_negative_axes"))
    }

    @Test
    fun test_slice_start_out_of_bounds() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_slice_start_out_of_bounds"))
    }
}
