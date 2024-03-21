package io.kinference.operators.conv

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ConvTest {
    private fun getTargetPath(dirName: String) = "conv/$dirName/"

    @Test
    fun test_conv_with_autopad_same() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_autopad_same"))
    }

    @Test
    fun test_conv_with_strides_and_asymmetric_padding() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_and_asymmetric_padding"))
    }

    @Test
    fun test_conv_with_strides_no_padding() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_no_padding"))
    }

    @Test
    fun test_conv_with_strides_padding() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_padding"))
    }

    @Test
    fun test_conv_feature_maps() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_feature_maps"))
    }

    @Test
    fun test_conv_with_bias() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_bias"))
    }

    @Test
    fun test_conv_with_groups() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups"))
    }

    @Test
    fun test_conv_with_groups_small() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups_small"))
    }

    @Test
    fun test_conv_with_dilations() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_dilations"))
    }

    @Test
    fun test_conv_with_5_dims() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_5_dims"))
    }
}
