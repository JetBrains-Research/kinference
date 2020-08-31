package io.kinference.operators.layer.normalization

import io.kinference.Utils
import org.junit.jupiter.api.Test

class SkipLayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/skip_layer_normalization/$dirName/"

    @Test
    fun `test skip layer normalization defaults`() {
        Utils.tensorTestRunner(getTargetPath("test_skip_layer_normalization"))
    }

    @Test
    fun `test skip layer normalization with bias`() {
        Utils.tensorTestRunner(getTargetPath("test_skip_layer_normalization_bias"))
    }
}
