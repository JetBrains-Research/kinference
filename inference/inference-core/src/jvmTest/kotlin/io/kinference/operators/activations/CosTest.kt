package io.kinference.operators.activations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CosTest {
    private fun getTargetPath(dirName: String) = "cos/$dirName/"

    @Test
    fun test_cos() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cos"))
    }

    @Test
    fun test_cos_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cos_example"))
    }
}
