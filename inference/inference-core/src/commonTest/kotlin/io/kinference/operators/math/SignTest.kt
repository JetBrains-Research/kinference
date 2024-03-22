package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class SignTest {
    private fun getTargetPath(dirName: String) = "sign/$dirName/"

    @Test
    fun test_sign() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sign"))
    }
}
