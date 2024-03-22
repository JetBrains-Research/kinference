package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AtanhTest {
    private fun getTargetPath(dirName: String) = "atanh/$dirName/"

    @Test
    fun test_atanh() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atanh"))
    }

    @Test
    fun test_atanh_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atanh_example"))
    }
}
