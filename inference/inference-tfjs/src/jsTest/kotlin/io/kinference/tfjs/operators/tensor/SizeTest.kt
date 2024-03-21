package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SizeTest {
    private fun getTargetPath(dirName: String) = "size/$dirName/"

    @Test
    fun test_size_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_size_example"))
    }

    @Test
    fun test_size() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_size"))
    }
}
