package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReshapeTest {
    private fun getTargetPath(dirName: String) = "reshape/$dirName/"

    @Test
    fun test_reshape_with_extended_dimensions()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_extended_dims"))
    }

    @Test
    fun test_reshape_with_negative_dimension()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_negative_dim"))
    }

    @Test
    fun test_reshape_with_negative_extended_dimensions()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_negative_extended_dims"))
    }

    @Test
    fun test_reshape_with_one_dimension()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_one_dim"))
    }

    @Test
    fun test_reshape_with_reduced_dimensions()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_reduced_dims"))
    }

    @Test
    fun test_reshape_with_all_reordered_dimensions()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_reordered_all_dims"))
    }

    @Test
    fun test_reshape_with_reordered_last_dimensions()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_reordered_last_dims"))
    }

    @Test
    fun test_reshape_with_zero_and_negative_dimension()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_zero_and_negative_dim"))
    }

    @Test
    fun test_reshape_zero_dim()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reshape_zero_dim"))
    }
}
