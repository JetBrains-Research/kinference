package org.jetbrains.research.kotlin.inference.operators.math

import org.jetbrains.research.kotlin.inference.Utils
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
