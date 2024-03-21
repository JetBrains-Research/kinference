package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class AtanhTest {
    private fun getTargetPath(dirName: String) = "atanh/$dirName/"

    @Test
    fun test_atanh() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atanh"))
    }

    @Test
    fun test_atanh_example() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_atanh_example"))
    }
}
