package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ModTest {
    private fun getTargetPath(dirName: String) = "mod/$dirName/"

    @Test
    fun test_mod_broadcast() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_broadcast"))
    }

    @Test
    fun test_mod_int64_fmod() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_int64_fmod"))
    }

    @Test
    fun test_mod_mixed_sign_float16() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_float16"))
    }

    @Test
    fun test_mod_mixed_sign_float32() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_float32"))
    }

    @Test
    fun test_mod_mixed_sign_float64() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_float64"))
    }

    @Test
    fun test_mod_mixed_sign_int8() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int8"))
    }

    @Test
    fun test_mod_mixed_sign_int16() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int16"))
    }

    @Test
    fun test_mod_mixed_sign_int32() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int32"))
    }

    @Test
    fun test_mod_mixed_sign_int64() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_mixed_sign_int64"))
    }

    @Test
    fun test_mod_uint8() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_uint8"))
    }

    @Test
    fun test_mod_uint16() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_uint16"))
    }

    @Test
    fun test_mod_uint32() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_uint32"))
    }

    @Test
    fun test_mod_uint64() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mod_uint64"))
    }
}
