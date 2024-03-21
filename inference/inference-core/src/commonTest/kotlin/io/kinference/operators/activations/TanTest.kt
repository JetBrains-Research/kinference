package io.kinference.operators.activations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class TanTest {
    private fun getTargetPath(dirName: String) = "tan/$dirName/"

    @Test
    fun test_tan_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tan_example"))
    }

    @Test
    fun test_tan() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_tan"))
    }
}
