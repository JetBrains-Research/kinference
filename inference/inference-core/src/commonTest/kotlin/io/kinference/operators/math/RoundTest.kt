package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class RoundTest {
    private fun getTargetPath(dirName: String) = "round/$dirName/"

    @Test
    fun test_round() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_round"))
    }
}
