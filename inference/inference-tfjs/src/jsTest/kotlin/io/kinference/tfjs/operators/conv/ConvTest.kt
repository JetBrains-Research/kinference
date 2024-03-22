package io.kinference.tfjs.operators.conv

import io.kinference.tfjs.runners.TFJSTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ConvTest {
    private fun getTargetPath(dirName: String) = "conv/$dirName/"

    @Test
    fun test_conv_with_autopad_same() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_autopad_same"))
    }

    @Test
    fun test_conv_with_strides_and_asymmetric_padding() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_and_asymmetric_padding"))
    }

    @Test
    fun test_conv_with_strides_no_padding() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_no_padding"))
    }

    @Test
    fun test_conv_with_strides_padding() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_padding"))
    }

    @Test
    fun test_conv_feature_maps() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_feature_maps"))
    }

    @Test
    fun test_conv_with_bias() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_bias"))
    }

    @Test
    fun test_conv_with_groups() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups"))
    }

    @Test
    fun test_conv_with_groups_small() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups_small"))
    }

    @Test
    fun test_conv_with_dilations() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_dilations"))
    }
}
