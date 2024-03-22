package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class AbsTest {
    private fun getTargetPath(dirName: String) = "abs/$dirName/"

    @Test
    fun test_abs_default() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_abs"))
    }
}
