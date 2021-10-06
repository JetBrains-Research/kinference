package io.kinference.tfjs.operators.math

import io.kinference.tfjs.runners.AccuracyRunner
import io.kinference.tfjs.utils.TestRunner
import kotlin.test.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "/mul/$dirName/"

    @Test
    fun test_mul()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_mul"))
    }

    @Test
    fun test_mul_with_broadcast()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun test_mul_defaults()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_mul_example"))
    }
}
