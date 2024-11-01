package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SqueezeTest {
    private fun getTargetPath(dirName: String) = "squeeze/$dirName/"

    @Test
    fun test_v1_squeeze() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("v1/test_squeeze"))
    }

    @Test
    fun test_v1_squeeze_with_negative_axes() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("v1/test_squeeze_negative_axes"))
    }
}
