package io.kinference.operators.activations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SinhTest {
    private fun getTargetPath(dirName: String) = "sinh/$dirName/"

    @Test
    fun test_sinh() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sinh"))
    }

    @Test
    fun test_sinh_example() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sinh_example"))
    }
}
