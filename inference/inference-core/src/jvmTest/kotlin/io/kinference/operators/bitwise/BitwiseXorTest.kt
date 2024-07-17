package io.kinference.operators.bitwise

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class BitwiseXorTest {
    private fun getTargetPath(dirName: String) = "bitwiseXor/$dirName/"

    @Test
    fun test_bitwise_xor_i16_3d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_xor_i16_3d"))
    }

    @Test
    fun test_bitwise_xor_i32_2d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_xor_i32_2d"))
    }

    @Test
    fun test_bitwise_xor_ui8_bcast_4v3d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_xor_ui8_bcast_4v3d"))
    }

    @Test
    fun test_bitwise_xor_ui64_bcast_3v1d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_xor_ui64_bcast_3v1d"))
    }
}
