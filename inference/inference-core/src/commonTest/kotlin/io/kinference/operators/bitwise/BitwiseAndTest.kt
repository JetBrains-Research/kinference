package io.kinference.operators.bitwise

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class BitwiseAndTest {
    private fun getTargetPath(dirName: String) = "bitwiseAnd/$dirName/"

    @Test
    fun test_bitwise_and_i16_3d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_and_i16_3d"))
    }

    @Test
    fun test_bitwise_and_i32_2d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_and_i32_2d"))
    }

    @Test
    fun test_bitwise_and_ui8_bcast_4v3d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_and_ui8_bcast_4v3d"))
    }

    @Test
    fun test_bitwise_and_ui64_bcast_3v1d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_and_ui64_bcast_3v1d"))
    }
}
