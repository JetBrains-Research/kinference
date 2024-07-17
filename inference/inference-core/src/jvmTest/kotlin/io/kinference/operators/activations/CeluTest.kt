package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class CeluTest {
    private fun getTargetPath(dirName: String) = "celu/$dirName/"

    @Test
    fun test_celu() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_celu"))
    }

    @Test
    fun test_celu_expanded() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_celu_expanded"))
    }
}
