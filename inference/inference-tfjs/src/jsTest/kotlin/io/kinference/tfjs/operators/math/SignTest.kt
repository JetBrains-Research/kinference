package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SignTest {
    private fun getTargetPath(dirName: String) = "sign/$dirName/"

    @Test
    fun test_sign() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_sign"))
    }
}
