package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class FloorTest {
    private fun getTargetPath(dirName: String) = "floor/$dirName/"

    @Test
    fun test_floor() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_floor"))
    }

    @Test
    fun test_floor_example() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_floor_example"))
    }
}
