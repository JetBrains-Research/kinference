package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ReciprocalTest {
    private fun getTargetPath(dirName: String) = "reciprocal/$dirName/"

    @Test
    fun test_reciprocal() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reciprocal"))
    }

    @Test
    fun test_reciprocal_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_reciprocal_example"))
    }
}
