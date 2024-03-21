package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AsinTest {
    private fun getTargetPath(dirName: String) = "asin/$dirName/"

    @Test
    fun test_asin() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asin"))
    }

    @Test
    fun test_asin_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_asin_example"))
    }
}
