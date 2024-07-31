package io.kinference.operators.activations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AcoshTest {
    private fun getTargetPath(dirName: String) = "acosh/$dirName/"

    @Test
    fun test_acosh() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_acosh"))
    }

    @Test
    fun test_acosh_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_acosh_example"))
    }
}
