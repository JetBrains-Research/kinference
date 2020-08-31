package io.kinference.operators.layer.normalization

import io.kinference.Utils
import org.junit.jupiter.api.Test

class LayerNormalizationTest {
    private fun getTargetPath(dirName: String) = "/layer_normalization/$dirName/"

    @Test
    fun `test layer normalization defaults`() {
        Utils.tensorTestRunner(getTargetPath("test_layer_normalization_0"))
    }

    @Test
    fun `test with negative axis`() {
        Utils.tensorTestRunner(getTargetPath("test_negate_axis"))
    }
}
