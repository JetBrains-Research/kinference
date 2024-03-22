package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GemmTest {
    private fun getTargetPath(dirName: String) = "gemm/$dirName/"

    @Test
    fun test_gemm_all_attributes() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_all_attributes"))
    }

    @Test
    fun test_gemm_alpha() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_alpha"))
    }

    @Test
    fun test_gemm_beta() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_beta"))
    }

    @Test
    fun test_gemm_default_matrix_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_matrix_bias"))
    }

    @Test
    fun test_gemm_default_no_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_no_bias"))
    }

    @Test
    fun test_gemm_default_scalar_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_scalar_bias"))
    }

    @Test
    fun test_gemm_default_single_elem_vector_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_single_elem_vector_bias"))
    }

    @Test
    fun test_gemm_default_vector_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_vector_bias"))
    }

    @Test
    fun test_gemm_default_zero_bias() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_default_zero_bias"))
    }

    @Test
    fun test_gemm_transposeA() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_transposeA"))
    }

    @Test
    fun test_gemm_transposeB() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_gemm_transposeB"))
    }
}
