package io.kinference.tfjs.operators.layer.normalization

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SkipLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "skip_layer_normalization/$dirName/"

    @Test
    fun test_skip_layer_normalization_defaults()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization"))
    }

    @Test
    fun test_skip_layer_normalization_with_bias()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization_bias"))
    }
}
