package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IsNaNTest {
    private fun getTargetPath(dirName: String) = "isnan/$dirName/"

    @Test
    fun test_isnan() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isnan"))
    }
}
