package io.kinference.operators.logical

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GreaterTest {
    private fun getTargetPath(dirName: String) = "greater/$dirName/"

    @Test
    fun test_greater() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_greater"))
    }

    @Test
    fun test_greater_with_broadcast() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_greater_bcast"))
    }
}
