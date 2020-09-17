package io.kinference.operators.quantization

import io.kinference.Utils
import org.junit.jupiter.api.Test

class TestDynamicQuantizeLinear {
    private fun getTargetPath(dirName: String) = "/dynamic_quantize_linear/$dirName/"

    @Test
    fun `test dynamic quantize linear default`() {
        Utils.tensorTestRunner(getTargetPath("test_dynamicquantizelinear"))
    }

    @Test
    fun `test dynamic quantize linear max adjusted`() {
        Utils.tensorTestRunner(getTargetPath("test_dynamicquantizelinear_max_adjusted"))
    }

    @Test
    fun `test dynamic quantize linear min adjusted`() {
        Utils.tensorTestRunner(getTargetPath("test_dynamicquantizelinear_min_adjusted"))
    }
}
