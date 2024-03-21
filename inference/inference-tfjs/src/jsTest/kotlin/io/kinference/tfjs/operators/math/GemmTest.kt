package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GemmTest {
    private fun getTargetPath(dirName: String) = "gemm/$dirName/"

    @Test
    fun test_gemm_all_attributes() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_all_attributes"))
    }

    @Test
    fun test_gemm_alpha() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_alpha"))
    }

    @Test
    fun test_gemm_beta() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_beta"))
    }

    @Test
    fun test_gemm_default_matrix_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_matrix_bias"))
    }

    @Test
    fun test_gemm_default_no_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_no_bias"))
    }

    @Test
    fun test_gemm_default_scalar_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_scalar_bias"))
    }

    @Test
    fun test_gemm_default_single_elem_vector_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_single_elem_vector_bias"))
    }

    @Test
    fun test_gemm_default_vector_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_vector_bias"))
    }

    @Test
    fun test_gemm_default_zero_bias() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_zero_bias"))
    }

    @Test
    fun test_gemm_transposeA() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_transposeA"))
    }

    @Test
    fun test_gemm_transposeB() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_transposeB"))
    }
}
