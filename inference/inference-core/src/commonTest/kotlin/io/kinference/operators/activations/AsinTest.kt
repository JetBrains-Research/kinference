package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AsinTest {
    private fun getTargetPath(dirName: String) = "asin/$dirName/"

    @Test
    fun test_asin() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asin"))
    }

    @Test
    fun test_asin_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asin_example"))
    }
}
