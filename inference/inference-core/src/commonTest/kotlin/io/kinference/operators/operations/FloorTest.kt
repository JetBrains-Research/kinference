package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class FloorTest {
    private fun getTargetPath(dirName: String) = "floor/$dirName/"

    @Test
    fun test_ceil() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_floor"))
    }

    @Test
    fun test_ceil_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_floor_example"))
    }
}
