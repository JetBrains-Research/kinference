package io.kinference.operators.math

import io.kinference.Utils
import org.junit.jupiter.api.Test

class GemmTest {
    private fun getTargetPath(dirName: String) = "/gemm/$dirName/"

    @Test
    fun `test gemm all attributes`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_all_attributes"))
    }

    @Test
    fun `test gemm alpha`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_alpha"))
    }

    @Test
    fun `test gemm beta`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_beta"))
    }

    @Test
    fun `test gemm default matrix bias`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_default_matrix_bias"))
    }

    @Test
    fun `test gemm default no bias`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_default_no_bias"))
    }

    @Test
    fun `test gemm default scalar bias`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_default_scalar_bias"))
    }

    @Test
    fun `test gemm default single elem vector bias`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_default_single_elem_vector_bias"))
    }

    @Test
    fun `test gemm default vector bias`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_default_vector_bias"))
    }

    @Test
    fun `test gemm default zero bias`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_default_zero_bias"))
    }

    @Test
    fun `test gemm transposeA`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_transposeA"))
    }

    @Test
    fun `test gemm transposeB`() {
        Utils.tensorTestRunner(getTargetPath("test_gemm_transposeB"))
    }
}
