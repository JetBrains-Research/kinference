package io.kinference.operators.activations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CosTest {
    private fun getTargetPath(dirName: String) = "cos/$dirName/"

    @Test
    fun test_cos() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cos"))
    }

    @Test
    fun test_cos_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_cos_example"))
    }
}
