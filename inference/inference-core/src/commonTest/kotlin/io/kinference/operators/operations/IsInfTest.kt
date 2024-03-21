package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class IsInfTest {
    private fun getTargetPath(dirName: String) = "isinf/$dirName/"

    @Test
    fun test_isinf() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isinf"))
    }

    @Test
    fun test_isinf_negative() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isinf_negative"))
    }

    @Test
    fun test_isinf_positive() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isinf_positive"))
    }
}
