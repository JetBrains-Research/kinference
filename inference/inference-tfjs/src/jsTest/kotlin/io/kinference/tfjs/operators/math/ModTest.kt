package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ModTest {
    private fun getTargetPath(dirName: String) = "mod/$dirName/"

    @Test
    fun test_mod_broadcast() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_broadcast"))
    }

    @Test
    fun test_mod_int64_fmod() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_int64_fmod"))
    }

    @Test
    fun test_mod_mixed_sign_float16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_float16"))
    }

    @Test
    fun test_mod_mixed_sign_float32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_float32"))
    }

    @Test
    fun test_mod_mixed_sign_float64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_float64"))
    }

    @Test
    fun test_mod_mixed_sign_int8() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int8"))
    }

    @Test
    fun test_mod_mixed_sign_int16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int16"))
    }

    @Test
    fun test_mod_mixed_sign_int32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int32"))
    }

    @Test
    fun test_mod_mixed_sign_int64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int64"))
    }

    @Test
    fun test_mod_uint8() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_uint8"))
    }

    @Test
    fun test_mod_uint16() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_uint16"))
    }

    @Test
    fun test_mod_uint32() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_uint32"))
    }

    @Test
    fun test_mod_uint64() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_mod_uint64"))
    }
}
