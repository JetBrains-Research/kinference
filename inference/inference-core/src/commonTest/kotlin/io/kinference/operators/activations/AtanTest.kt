package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AtanTest {
    private fun getTargetPath(dirName: String) = "atan/$dirName/"

    @Test
    fun test_atan() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atan"))
    }

    @Test
    fun test_atan_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atan_example"))
    }
}
