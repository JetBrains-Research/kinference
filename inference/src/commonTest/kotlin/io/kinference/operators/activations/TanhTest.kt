package io.kinference.operators.activations

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class TanhTest {
    private fun getTargetPath(dirName: String) = "/tanh/$dirName/"

    @Test
    fun test_tanh_example()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_tanh_example"))
    }

    @Test
    fun test_tanh()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_tanh"))
    }

    @Test
    fun test_tanh_scalar()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_tanh_scalar"))
    }
}
