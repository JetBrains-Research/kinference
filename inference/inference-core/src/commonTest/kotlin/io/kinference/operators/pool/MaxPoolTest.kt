package io.kinference.operators.pool

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MaxPoolTest {
    private fun getTargetPath(dirName: String) = "maxpool/$dirName/"

    @Test
    fun test_maxpool_1d_default() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_1d_default"))
    }

    @Test
    fun test_maxpool_2d_ceil() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_ceil"))
    }

    @Test
    fun test_maxpool_2d_default() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_default"))
    }

    @Test
    fun test_maxpool_2d_dilations() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_dilations"))
    }

    @Test
    fun test_maxpool_2d_pads() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_pads"))
    }

    @Test
    fun test_maxpool_2d_precomputed_pads() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_precomputed_pads"))
    }

    @Test
    fun test_maxpool_2d_precomputed_strides() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_precomputed_strides"))
    }

    @Test
    fun test_maxpool_2d_same_lower() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_same_lower"))
    }

    @Test
    fun test_maxpool_2d_same_upper() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_same_upper"))
    }

    @Test
    fun test_maxpool_2d_strides() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_strides"))
    }

    @Test
    fun test_maxpool_2d_uint8() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_uint8"))
    }

    @Test
    fun test_maxpool_3d_default() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_3d_default"))
    }

    @Test
    fun test_maxpool_with_argmax_2d_precomputed_pads() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_with_argmax_2d_precomputed_pads"))
    }

    @Test
    fun test_maxpool_with_argmax_2d_precomputed_strides() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_with_argmax_2d_precomputed_strides"))
    }

    @Test
    fun test_maxpool_3d_argmax_st_ord_1() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_3d_argmax_st_ord_1"))
    }
}
