package io.kinference.operators.logical

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LessTest {
    private fun getTargetPath(dirName: String) = "less/$dirName/"

    @Test
    fun test_less() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_less"))
    }

    @Test
    fun test_less_with_broadcast() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_less_bcast"))
    }
}
