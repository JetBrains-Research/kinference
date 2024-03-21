package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class FloorTest {
    private fun getTargetPath(dirName: String) = "floor/$dirName/"

    @Test
    fun test_floor() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_floor"))
    }

    @Test
    fun test_floor_example() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_floor_example"))
    }
}
