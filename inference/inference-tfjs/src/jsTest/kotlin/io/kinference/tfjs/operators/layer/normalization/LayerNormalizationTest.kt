package io.kinference.tfjs.operators.layer.normalization

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "layer_normalization/$dirName/"

    @Test
    fun test_layer_normalization_defaults()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_layer_normalization_0"))
    }

//    @Test
    fun test_with_negative_axis()  = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_negate_axis"))
    }
}
