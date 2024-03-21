package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CeilTest {
    private fun getTargetPath(dirName: String) = "ceil/$dirName/"

    @Test
    fun test_ceil() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_ceil"))
    }

    @Test
    fun test_ceil_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_ceil_example"))
    }
}
