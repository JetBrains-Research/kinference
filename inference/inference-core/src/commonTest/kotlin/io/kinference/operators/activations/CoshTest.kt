package io.kinference.operators.activations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CoshTest {
    private fun getTargetPath(dirName: String) = "cosh/$dirName/"

    @Test
    fun test_cosh() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cosh"))
    }

    @Test
    fun test_cosh_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cosh_example"))
    }
}
