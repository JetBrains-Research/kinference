package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GemmTest {
    private fun getTargetPath(dirName: String) = "gemm/$dirName/"

    @Test
    fun test_gemm_all_attributes() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_all_attributes"))
    }

    @Test
    fun test_gemm_alpha() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_alpha"))
    }

    @Test
    fun test_gemm_beta() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_beta"))
    }

    @Test
    fun test_gemm_default_matrix_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_matrix_bias"))
    }

    @Test
    fun test_gemm_default_no_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_no_bias"))
    }

    @Test
    fun test_gemm_default_scalar_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_scalar_bias"))
    }

    @Test
    fun test_gemm_default_single_elem_vector_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_single_elem_vector_bias"))
    }

    @Test
    fun test_gemm_default_vector_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_vector_bias"))
    }

    @Test
    fun test_gemm_default_zero_bias() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_zero_bias"))
    }

    @Test
    fun test_gemm_transposeA() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_transposeA"))
    }

    @Test
    fun test_gemm_transposeB() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gemm_transposeB"))
    }
}
