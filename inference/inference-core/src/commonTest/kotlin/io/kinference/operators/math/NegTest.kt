package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class NegTest {
    private fun getTargetPath(dirName: String) = "neg/$dirName/"

    @Test
    fun test_neg() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_neg"))
    }

    @Test
    fun test_neg_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_neg_example"))
    }
}
