package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "shape/$dirName/"

    @Test
    fun test_shape() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun test_shape_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_example"))
    }

    @Test
    fun test_shape_clip_start() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_clip_start"))
    }

    @Test
    fun test_shape_start_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_start_1"))
    }

    @Test
    fun test_shape_start_negative_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_start_negative_1"))
    }

    @Test
    fun test_shape_clip_end() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_clip_end"))
    }

    @Test
    fun test_shape_end_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_end_1"))
    }

    @Test
    fun test_shape_end_negative_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_end_negative_1"))
    }

    @Test
    fun test_shape_start_1_end_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_start_1_end_2"))
    }

    @Test
    fun test_shape_start_1_end_negative_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_shape_start_1_end_negative_1"))
    }
}
