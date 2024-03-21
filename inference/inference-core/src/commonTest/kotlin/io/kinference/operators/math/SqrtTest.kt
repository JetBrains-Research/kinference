package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SqrtTest {
    private fun getTargetPath(dirName: String) = "sqrt/$dirName/"

    @Test
    fun test_sqrt() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sqrt"))
    }

    @Test
    fun test_sqrt_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sqrt_example"))
    }
}
