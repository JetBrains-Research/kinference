package io.kinference.operators.quantization

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class TestDynamicQuantizeLinear {
    private fun getTargetPath(dirName: String) = "/dynamic_quantize_linear/$dirName/"

    @Test
    fun `test dynamic quantize linear default`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear"))
    }

    @Test
    fun `test dynamic quantize linear max adjusted`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_max_adjusted"))
    }

    @Test
    fun `test dynamic quantize linear min adjusted`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_dynamicquantizelinear_min_adjusted"))
    }
}
