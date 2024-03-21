package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class UnsqueezeTest {
    private fun getTargetPath(dirName: String) = "unsqueeze/$dirName/"

    @Test
    fun test_unsqueeze_axis_0()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_0"))
    }

    @Test
    fun test_unsqueeze_axis_1()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_1"))
    }

    @Test
    fun test_unsqueeze_axis_2()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_2"))
    }

    @Test
    fun test_unsqueeze_axis_3()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_3"))
    }

    @Test
    fun test_unsqueeze_with_negative_axes()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_negative_axes"))
    }

    @Test
    fun test_unsqueeze_three_axes()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_three_axes"))
    }

    @Test
    fun test_unsqueeze_two_axes()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_two_axes"))
    }

    @Test
    fun test_unsqueeze_unsorted_axes()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_unsorted_axes"))
    }


    @Test
    fun test_unsqueeze_axis_0_v13() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_0_v13"))
    }

    @Test
    fun test_unsqueeze_axis_1_v13() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_1_v13"))
    }

    @Test
    fun test_unsqueeze_axis_2_v13() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_2_v13"))
    }

    @Test
    fun test_unsqueeze_with_negative_axes_v13() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_negative_axes_v13"))
    }

    @Test
    fun test_unsqueeze_three_axes_v13() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_three_axes_v13"))
    }

    @Test
    fun test_unsqueeze_two_axes_v13() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_two_axes_v13"))
    }

    @Test
    fun test_unsqueeze_unsorted_axes_v13() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_unsorted_axes_v13"))
    }
}
