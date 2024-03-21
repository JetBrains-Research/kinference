package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AcosTest {
    private fun getTargetPath(dirName: String) = "acos/$dirName/"

    @Test
    fun test_acos() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_acos"))
    }

    @Test
    fun test_acos_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_acos_example"))
    }
}
