package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class EqualTest {
    private fun getTargetPath(dirName: String) = "equal/$dirName/"

    @Test
    fun test_equal() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_equal"))
    }

    @Test
    fun test_equal_with_broadcast() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_equal_bcast"))
    }
}
