package io.kinference.operators.layer.normalization

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SkipLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "skip_layer_normalization/$dirName/"

    @Test
    fun test_skip_layer_normalization_defaults() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization"))
    }

    @Test
    fun test_skip_layer_normalization_with_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization_bias"))
    }
}
