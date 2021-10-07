package io.kinference.operators.operations

import io.kinference.runners.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ConcatTest {
    private fun getTargetPath(dirName: String) = "/concat/$dirName/"

    @Test
    fun test_concat_1D_axis_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_1d_axis_0"))
    }

    @Test
    fun test_concat_1D_axis_neg_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_1d_axis_negative_1"))
    }

    @Test
    fun test_concat_2D_axis_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_0"))
    }

    @Test
    fun test_concat_2D_axis_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_1"))
    }

    @Test
    fun test_concat_2D_axis_neg_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun test_concat_2D_axis_neg_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun test_concat_3D_axis_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_0"))
    }

    @Test
    fun test_concat_3D_axis_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_1"))
    }

    @Test
    fun test_concat_3D_axis_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_2"))
    }

    @Test
    fun test_concat_3D_axis_neg_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun test_concat_3D_axis_neg_2() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun test_concat_3D_axis_neg_3() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_negative_3"))
    }
}
