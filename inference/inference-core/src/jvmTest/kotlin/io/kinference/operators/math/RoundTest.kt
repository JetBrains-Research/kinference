package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class RoundTest {
    private fun getTargetPath(dirName: String) = "round/$dirName/"

    @Test
    fun test_round() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_round"))
    }
}
