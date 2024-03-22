package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IsInfTest {
    private fun getTargetPath(dirName: String) = "isinf/$dirName/"

    @Test
    fun test_isinf() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isinf"))
    }

    @Test
    fun test_isinf_negative() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isinf_negative"))
    }

    @Test
    fun test_isinf_positive() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isinf_positive"))
    }
}
