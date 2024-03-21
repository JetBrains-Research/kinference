package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "shape/$dirName/"

    @Test
    fun test_shape() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun test_shape_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_example"))
    }

    @Test
    fun test_shape_clip_start() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_clip_start"))
    }

    @Test
    fun test_shape_start_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_start_1"))
    }

    @Test
    fun test_shape_start_negative_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_start_negative_1"))
    }

    @Test
    fun test_shape_clip_end() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_clip_end"))
    }

    @Test
    fun test_shape_end_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_end_1"))
    }

    @Test
    fun test_shape_end_negative_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_end_negative_1"))
    }

    @Test
    fun test_shape_start_1_end_2() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_start_1_end_2"))
    }

    @Test
    fun test_shape_start_1_end_negative_1() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_shape_start_1_end_negative_1"))
    }
}
