package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class NonZeroTest {
    private fun getTargetPath(dirName: String) = "nonzero/$dirName/"

    @Test
    fun test_nonzero() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_nonzero"))
    }
}
