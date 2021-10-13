package io.kinference.operators.layer.normalization

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SkipLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/skip_layer_normalization/$dirName/"

    @Test
    fun test_skip_layer_normalization_defaults() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization"))
    }

    @Test
    fun test_skip_layer_normalization_with_bias() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization_bias"))
    }
}
