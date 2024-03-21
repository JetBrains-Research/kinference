package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class GreaterTest {
    private fun getTargetPath(dirName: String) = "greater/$dirName/"

    @Test
    fun test_greater() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_greater"))
    }

    @Test
    fun test_greater_with_broadcast() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_greater_bcast"))
    }
}
