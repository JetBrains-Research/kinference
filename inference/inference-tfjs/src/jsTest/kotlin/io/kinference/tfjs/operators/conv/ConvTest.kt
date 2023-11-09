package io.kinference.tfjs.operators.conv

import io.kinference.tfjs.runners.TFJSTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ConvTest {
    private fun getTargetPath(dirName: String) = "conv/$dirName/"

    // pass
    @Test
    fun test_conv_with_autopad_same() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_autopad_same"))
    }

    // pass
    @Test
    fun test_conv_with_strides_and_asymmetric_padding() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_and_asymmetric_padding"))
    }

    // bad
    @Test
    fun test_conv_with_strides_no_padding() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_no_padding"))
    }

    // bad
    @Test
    fun test_conv_with_strides_padding() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_padding"))
    }

    // pass
    @Test
    fun test_conv_feature_maps() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_feature_maps"))
    }

    // pass, but output is not the same as for kinference core conv operator
    @Test
    fun test_conv_with_bias() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_bias"))
    }

    // pass
    @Test
    fun test_conv_with_groups() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups"))
    }

    // pass
    @Test
    fun test_conv_with_groups_small() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups_small"))
    }

    // pass
    @Test
    fun test_conv_with_dilations() = TestRunner.runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_dilations"))
    }

    // This test is for tensors with rank 7, tfjs doesn't support it
//    @Test
//    fun test_conv_with_5_dims() = TestRunner.runTest {
//        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_conv_with_5_dims"))
//    }
}
