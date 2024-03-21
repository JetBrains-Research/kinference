package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class RoundTest {
    private fun getTargetPath(dirName: String) = "round/$dirName/"

    @Test
    fun test_round() = runTest {
        TFJSTestEngine.TFJSAccuracyRunner.runFromResources(getTargetPath("test_round"))
    }
}
