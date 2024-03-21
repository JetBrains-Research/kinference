package io.kinference.operators.activations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AcoshTest {
    private fun getTargetPath(dirName: String) = "acosh/$dirName/"

    @Test
    fun test_acosh() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_acosh"))
    }

    @Test
    fun test_acosh_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_acosh_example"))
    }
}
