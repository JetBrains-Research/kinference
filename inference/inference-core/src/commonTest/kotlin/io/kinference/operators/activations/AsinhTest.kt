package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AsinhTest {
    private fun getTargetPath(dirName: String) = "asinh/$dirName/"

    @Test
    fun test_asinh() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asinh"))
    }

    @Test
    fun test_asinh_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asinh_example"))
    }
}
