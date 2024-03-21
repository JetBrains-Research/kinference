package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TanhTest {
    private fun getTargetPath(dirName: String) = "tanh/$dirName/"

    @Test
    fun test_tanh_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tanh_example"))
    }

    @Test
    fun test_tanh() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tanh"))
    }

    @Test
    fun test_tanh_scalar() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_tanh_scalar"))
    }
}
