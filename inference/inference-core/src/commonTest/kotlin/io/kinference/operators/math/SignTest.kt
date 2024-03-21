package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class SignTest {
    private fun getTargetPath(dirName: String) = "sign/$dirName/"

    @Test
    fun test_sign() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_sign"))
    }
}
