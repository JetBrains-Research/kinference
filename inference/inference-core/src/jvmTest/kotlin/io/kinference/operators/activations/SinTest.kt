package io.kinference.operators.activations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SinTest {
    private fun getTargetPath(dirName: String) = "sin/$dirName/"

    @Test
    fun test_sin() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sin"))
    }

    @Test
    fun test_sin_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sin_example"))
    }
}
