package io.kinference.tfjs.operators.layer.normalization

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class LayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/layer_normalization/$dirName/"

    @Test
    fun test_layer_normalization_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_layer_normalization_0"))
    }

//    @Test
    fun test_with_negative_axis()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_negate_axis"))
    }
}
