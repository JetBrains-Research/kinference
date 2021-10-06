package io.kinference.operators.layer.normalization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
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
