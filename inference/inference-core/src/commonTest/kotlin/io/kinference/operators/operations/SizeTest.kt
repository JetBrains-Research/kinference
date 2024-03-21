package io.kinference.operators.operations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SizeTest {
    private fun getTargetPath(dirName: String) = "size/$dirName/"

    @Test
    fun test_size_example() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_size_example"))
    }

    @Test
    fun test_size() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_size"))
    }
}
