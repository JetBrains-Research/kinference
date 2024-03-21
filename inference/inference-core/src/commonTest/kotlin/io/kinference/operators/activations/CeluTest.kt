package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CeluTest {
    private fun getTargetPath(dirName: String) = "celu/$dirName/"

    @Test
    fun test_celu() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_celu"))
    }

    @Test
    fun test_celu_expanded() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_celu_expanded"))
    }
}
