package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IsInfTest {
    private fun getTargetPath(dirName: String) = "isinf/$dirName/"

    @Test
    fun test_isinf() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isinf"))
    }

    @Test
    fun test_isinf_negative() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isinf_negative"))
    }

    @Test
    fun test_isinf_positive() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_isinf_positive"))
    }
}
