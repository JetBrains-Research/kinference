package io.kinference.operators.operations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class ConcatTest {
    private fun getTargetPath(dirName: String) = "/concat/$dirName/"

    @Test
    fun `test concat 1D (axis=0)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_1d_axis_0"))
    }

    @Test
    fun `test concat 1D (axis=-1)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_1d_axis_negative_1"))
    }

    @Test
    fun `test concat 2D (axis=0)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_0"))
    }

    @Test
    fun `test concat 2D (axis=1)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_1"))
    }

    @Test
    fun `test concat 2D (axis=-1)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun `test concat 2D (axis=-2)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun `test concat 3D (axis=0)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_0"))
    }

    @Test
    fun `test concat 3D (axis=1)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_1"))
    }

    @Test
    fun `test concat 3D (axis=2)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_2"))
    }

    @Test
    fun `test concat 3D (axis=-1)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_1"))
    }

    @Test
    fun `test concat 3D (axis=-2)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_2d_axis_negative_2"))
    }

    @Test
    fun `test concat 3D (axis=-3)`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_concat_3d_axis_negative_3"))
    }
}
