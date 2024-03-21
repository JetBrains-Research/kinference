package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "mul/$dirName/"

    @Test
    fun test_mul() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mul"))
    }

    @Test
    fun test_mul_with_broadcast() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun test_mul_defaults() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_mul_example"))
    }
}
