package io.kinference.tfjs.operators.conv

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ConvTest {
    private fun getTargetPath(dirName: String) = "conv/$dirName/"

    @Test
    fun test_conv_with_autopad_same() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_autopad_same"))
    }

    @Test
    fun test_conv_with_strides_and_asymmetric_padding() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_and_asymmetric_padding"))
    }

    @Test
    fun test_conv_with_strides_no_padding() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_no_padding"))
    }

    @Test
    fun test_conv_with_strides_padding() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_padding"))
    }

    @Test
    fun test_conv_feature_maps() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_feature_maps"))
    }

    @Test
    fun test_conv_with_bias() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_bias"))
    }

    @Test
    fun test_conv_with_groups() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups"))
    }

    @Test
    fun test_conv_with_groups_small() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups_small"))
    }

    @Test
    fun test_conv_with_dilations() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_dilations"))
    }
}
