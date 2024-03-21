package io.kinference.operators.activations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CoshTest {
    private fun getTargetPath(dirName: String) = "cosh/$dirName/"

    @Test
    fun test_cosh() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cosh"))
    }

    @Test
    fun test_cosh_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cosh_example"))
    }
}
