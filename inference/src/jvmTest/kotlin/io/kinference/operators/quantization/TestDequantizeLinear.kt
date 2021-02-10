package io.kinference.operators.quantization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class TestDequantizeLinear {
    private fun getTargetPath(dirName: String) = "/dequantize_linear/$dirName/"

    @Test
    fun `test linear dequantization defaults`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear"))
    }

    @Test
    fun `test linear dequantization per tensor`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_tensor"))
    }

    @Test
    fun `test linear dequantization per axis=1`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_per_axis_1"))
    }

    @Test
    fun `test linear dequantization per axis`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dequantizelinear_axis"))
    }
}
