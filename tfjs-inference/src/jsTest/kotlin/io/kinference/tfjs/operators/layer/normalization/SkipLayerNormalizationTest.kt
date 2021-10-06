package io.kinference.tfjs.operators.layer.normalization

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class SkipLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/skip_layer_normalization/$dirName/"

    @Test
    fun test_skip_layer_normalization_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization"))
    }

    @Test
    fun test_skip_layer_normalization_with_bias()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization_bias"))
    }
}
