package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SplitTest {
    private fun getTargetPath(dirName: String) = "split/$dirName/"

    @Test
    fun test_split_into_equal_parts_1D() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_1d"))
    }

    @Test
    fun test_split_into_equal_parts_2D() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun test_split_into_equal_parts_default_axis() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_default_axis"))
    }

    @Test
    fun test_split_into_variable_parts_1D() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_split_variable_parts_1d"))
    }

    @Test
    fun test_split_into_variable_parts_2D() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_split_equal_parts_2d"))
    }

    @Test
    fun test_split_into_variable_parts_default_axis() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_split_variable_parts_default_axis"))
    }

    @Test
    fun test_split_into_zero_size_splits() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_split_zero_size_splits"))
    }
}
