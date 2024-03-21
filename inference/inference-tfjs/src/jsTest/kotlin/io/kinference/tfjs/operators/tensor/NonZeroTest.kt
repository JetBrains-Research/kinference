package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class NonZeroTest {
    private fun getTargetPath(dirName: String) = "nonzero/$dirName/"

    @Test
    fun test_nonzero() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_nonzero"))
    }
}
