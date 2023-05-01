package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AsinhTest {
    private fun getTargetPath(dirName: String) = "asinh/$dirName/"

    @Test
    fun test_asinh() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asinh"))
    }

    @Test
    fun test_asinh_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asinh_example"))
    }
}
