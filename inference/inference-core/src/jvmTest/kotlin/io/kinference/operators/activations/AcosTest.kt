package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AcosTest {
    private fun getTargetPath(dirName: String) = "acos/$dirName/"

    @Test
    fun test_acos() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_acos"))
    }

    @Test
    fun test_acos_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_acos_example"))
    }
}
