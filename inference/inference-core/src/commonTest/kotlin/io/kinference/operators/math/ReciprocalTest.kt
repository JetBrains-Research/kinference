package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ReciprocalTest {
    private fun getTargetPath(dirName: String) = "reciprocal/$dirName/"

    @Test
    fun test_reciprocal() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reciprocal"))
    }

    @Test
    fun test_reciprocal_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reciprocal_example"))
    }
}
