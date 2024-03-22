package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SqrtTest {
    private fun getTargetPath(dirName: String) = "sqrt/$dirName/"

    @Test
    fun test_sqrt() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sqrt"))
    }

    @Test
    fun test_sqrt_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sqrt_example"))
    }
}
