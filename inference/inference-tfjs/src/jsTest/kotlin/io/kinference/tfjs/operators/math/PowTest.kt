package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class PowTest {
    private fun getTargetPath(dirName: String) = "pow/$dirName/"

    @Test
    fun test_pow() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow"), delta = 2e-3)
    }

    @Test
    fun test_pow_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_example"))
    }

    @Test
    fun test_pow_bcast_array() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_bcast_array"))
    }

    @Test
    fun test_pow_bcast_scalar() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_bcast_scalar"))
    }

    @Test
    fun test_pow_types_float32_int32() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_int32"))
    }

    @Test
    fun test_pow_types_float32_int64() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_int64"))
    }

    @Test
    fun test_pow_types_float32_uint32() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_uint32"))
    }

    @Test
    fun test_pow_types_float32_uint64() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_uint64"))
    }

    @Test
    fun test_pow_types_int32_float32() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int32_float32"))
    }

    @Test
    fun test_pow_types_int32_int32() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int32_int32"))
    }

    @Test
    fun test_pow_types_int64_float32() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int64_float32"))
    }

    @Test
    fun test_pow_types_int64_int64() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int64_int64"))
    }
}
