package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class FloorTest {
    private fun getTargetPath(dirName: String) = "floor/$dirName/"

    @Test
    fun test_ceil() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_floor"))
    }

    @Test
    fun test_ceil_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_floor_example"))
    }
}
