package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReciprocalTest {
    private fun getTargetPath(dirName: String) = "reciprocal/$dirName/"

    @Test
    fun test_reciprocal() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reciprocal"))
    }

    @Test
    fun test_reciprocal_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_reciprocal_example"))
    }
}
