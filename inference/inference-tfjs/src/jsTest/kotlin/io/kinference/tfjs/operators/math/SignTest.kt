package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SignTest {
    private fun getTargetPath(dirName: String) = "sign/$dirName/"

    @Test
    fun test_sign() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sign"))
    }
}
