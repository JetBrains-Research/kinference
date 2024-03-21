package io.kinference.operators.bitwise

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class BitShiftTest {
    private fun getTargetPath(dirName: String) = "bitShift/$dirName/"

    @Test
    fun test_bitshift_left_uint8() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_left_uint8"))
    }

    @Test
    fun test_bitshift_left_uint16() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_left_uint16"))
    }

    @Test
    fun test_bitshift_left_uint32() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_left_uint32"))
    }

    @Test
    fun test_bitshift_left_uint64() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_left_uint64"))
    }

    @Test
    fun test_bitshift_right_uint8() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_right_uint8"))
    }

    @Test
    fun test_bitshift_right_uint16() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_right_uint16"))
    }

    @Test
    fun test_bitshift_right_uint32() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_right_uint32"))
    }

    @Test
    fun test_bitshift_right_uint64() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitshift_right_uint64"))
    }
}
