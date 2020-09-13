package io.kinference.operators.quantization

import io.kinference.Utils
import org.junit.jupiter.api.Test

class TestDequantizeLinear {
    private fun getTargetPath(dirName: String) = "/dequantize_linear/$dirName/"

    @Test
    fun `test linear dequantization defaults`() {
        Utils.tensorTestRunner(getTargetPath("test_dequantizelinear"))
    }

    @Test
    fun `test linear dequantization per tensor`() {
        Utils.tensorTestRunner(getTargetPath("test_dequantizelinear_per_tensor"))
    }

    @Test
    fun `test linear dequantization per axis=1`() {
        Utils.tensorTestRunner(getTargetPath("test_dequantizelinear_per_axis_1"))
    }

    @Test
    fun `test linear dequantization per axis`() {
        Utils.tensorTestRunner(getTargetPath("test_dequantizelinear_axis"))
    }
}
