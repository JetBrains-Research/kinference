package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class NonZeroTest {
    private fun getTargetPath(dirName: String) = "nonzero/$dirName/"

    @Test
    fun test_nonzero() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_nonzero"))
    }
}
