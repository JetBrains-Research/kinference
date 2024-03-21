package io.kinference.operators.pool

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MaxPoolTest {
    private fun getTargetPath(dirName: String) = "maxpool/$dirName/"

    @Test
    fun test_maxpool_1d_default() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_1d_default"))
    }

    @Test
    fun test_maxpool_2d_ceil() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_ceil"))
    }

    @Test
    fun test_maxpool_2d_default() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_default"))
    }

    @Test
    fun test_maxpool_2d_dilations() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_dilations"))
    }

    @Test
    fun test_maxpool_2d_pads() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_pads"))
    }

    @Test
    fun test_maxpool_2d_precomputed_pads() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_precomputed_pads"))
    }

    @Test
    fun test_maxpool_2d_precomputed_strides() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_precomputed_strides"))
    }

    @Test
    fun test_maxpool_2d_same_lower() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_same_lower"))
    }

    @Test
    fun test_maxpool_2d_same_upper() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_same_upper"))
    }

    @Test
    fun test_maxpool_2d_strides() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_strides"))
    }

    @Test
    fun test_maxpool_2d_uint8() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_2d_uint8"))
    }

    @Test
    fun test_maxpool_3d_default() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_3d_default"))
    }

    @Test
    fun test_maxpool_with_argmax_2d_precomputed_pads() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_with_argmax_2d_precomputed_pads"))
    }

    @Test
    fun test_maxpool_with_argmax_2d_precomputed_strides() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_with_argmax_2d_precomputed_strides"))
    }

    @Test
    fun test_maxpool_3d_argmax_st_ord_1() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_maxpool_3d_argmax_st_ord_1"))
    }
}
