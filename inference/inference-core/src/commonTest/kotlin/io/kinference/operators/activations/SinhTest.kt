package io.kinference.operators.activations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SinhTest {
    private fun getTargetPath(dirName: String) = "sinh/$dirName/"

    @Test
    fun test_sinh() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sinh"))
    }

    @Test
    fun test_sinh_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sinh_example"))
    }
}
