package io.kinference.operators.conv

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ConvTest {
    private fun getTargetPath(dirName: String) = "conv/$dirName/"

    @Test
    fun test_conv_with_autopad_same() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_autopad_same"))
    }

    @Test
    fun test_conv_with_strides_and_asymmetric_padding() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_and_asymmetric_padding"))
    }

    @Test
    fun test_conv_with_strides_no_padding() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_no_padding"))
    }

    @Test
    fun test_conv_with_strides_padding() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_strides_padding"))
    }

    @Test
    fun test_conv_feature_maps() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_feature_maps"))
    }

    @Test
    fun test_conv_with_bias() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_bias"))
    }

    @Test
    fun test_conv_with_groups() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups"))
    }

    @Test
    fun test_conv_with_groups_small() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_groups_small"))
    }

    @Test
    fun test_conv_with_dilations() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_dilations"))
    }

    @Test
    fun test_conv_with_5_dims() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_conv_with_5_dims"))
    }
}
