package io.kinference.operators.bitwise

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class BitwiseOrTest {
    private fun getTargetPath(dirName: String) = "bitwiseOr/$dirName/"

    @Test
    fun test_bitwise_or_i16_4d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_i16_4d"))
    }

    @Test
    fun test_bitwise_or_i32_2d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_i32_2d"))
    }

    @Test
    fun test_bitwise_or_ui8_bcast_4v3d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_ui8_bcast_4v3d"))
    }

    @Test
    fun test_bitwise_or_ui64_bcast_3v1d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_or_ui64_bcast_3v1d"))
    }
}
