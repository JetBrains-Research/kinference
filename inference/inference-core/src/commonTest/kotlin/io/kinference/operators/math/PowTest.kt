package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class PowTest {
    private fun getTargetPath(dirName: String) = "pow/$dirName/"

    @Test
    fun test_pow() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow"))
    }

    @Test
    fun test_pow_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_example"))
    }

    @Test
    fun test_pow_bcast_array() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_bcast_array"))
    }

    @Test
    fun test_pow_bcast_scalar() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_bcast_scalar"))
    }

    @Test
    fun test_pow_types_float32_int32() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_int32"))
    }

    /*@Test
    fun test_pow_types_float32_int64() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_int64"))
    }*/

    @Test
    fun test_pow_types_float32_uint32() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_uint32"))
    }

    /*@Test
    fun test_pow_types_float32_uint64() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_float32_uint64"))
    }*/

    @Test
    fun test_pow_types_int32_float32() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int32_float32"))
    }

    @Test
    fun test_pow_types_int32_int32() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int32_int32"))
    }

    @Test
    fun test_pow_types_int64_float32() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int64_float32"))
    }

    /*@Test
    fun test_pow_types_int64_int64() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_pow_types_int64_int64"))
    }*/
}
