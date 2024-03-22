package io.kinference.operators.bitwise

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class BitwiseOrTest {
    private fun getTargetPath(dirName: String) = "bitwiseOr/$dirName/"

    @Test
    fun test_bitwise_or_i16_4d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_i16_4d"))
    }

    @Test
    fun test_bitwise_or_i32_2d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_i32_2d"))
    }

    @Test
    fun test_bitwise_or_ui8_bcast_4v3d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_ui8_bcast_4v3d"))
    }

    @Test
    fun test_bitwise_or_ui64_bcast_3v1d() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_ui64_bcast_3v1d"))
    }
}
