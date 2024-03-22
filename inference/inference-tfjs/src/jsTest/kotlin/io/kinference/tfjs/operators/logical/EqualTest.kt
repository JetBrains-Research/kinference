package io.kinference.tfjs.operators.logical

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class EqualTest {
    private fun getTargetPath(dirName: String) = "equal/$dirName/"

    @Test
    fun test_equal() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_equal"))
    }

    @Test
    fun test_equal_with_broadcast() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_equal_bcast"))
    }
}
