package io.kinference.operators.layer.normalization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class SkipLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/skip_layer_normalization/$dirName/"

    @Test
    fun `test skip layer normalization defaults`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization"))
    }

    @Test
    fun `test skip layer normalization with bias`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_skip_layer_normalization_bias"))
    }
}
