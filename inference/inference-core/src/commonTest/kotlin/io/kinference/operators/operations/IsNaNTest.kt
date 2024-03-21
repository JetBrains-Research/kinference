package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IsNaNTest {
    private fun getTargetPath(dirName: String) = "isnan/$dirName/"

    @Test
    fun test_isnan() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isnan"))
    }
}
