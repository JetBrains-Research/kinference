package io.kinference.operators.bitwise

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class BitwiseNotTest {
    private fun getTargetPath(dirName: String) = "bitwiseNot/$dirName/"

    @Test
    fun test_bitwise_not_2d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_not_2d"))
    }

    @Test
    fun test_bitwise_not_3d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_not_3d"))
    }

    @Test
    fun test_bitwise_not_4d() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_bitwise_not_4d"))
    }
}
