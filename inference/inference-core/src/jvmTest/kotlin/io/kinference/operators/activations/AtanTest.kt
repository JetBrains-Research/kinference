package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AtanTest {
    private fun getTargetPath(dirName: String) = "atan/$dirName/"

    @Test
    fun test_atan() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atan"))
    }

    @Test
    fun test_atan_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atan_example"))
    }
}
