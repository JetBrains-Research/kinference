package io.kinference.operators.activations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SinTest {
    private fun getTargetPath(dirName: String) = "sin/$dirName/"

    @Test
    fun test_sin() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sin"))
    }

    @Test
    fun test_sin_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sin_example"))
    }
}
