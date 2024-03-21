package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class NegTest {
    private fun getTargetPath(dirName: String) = "neg/$dirName/"

    @Test
    fun test_neg() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_neg"))
    }

    @Test
    fun test_neg_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_neg_example"))
    }
}
