package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class IsInfTest {
    private fun getTargetPath(dirName: String) = "isinf/$dirName/"

    @Test
    fun test_isinf() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isinf"))
    }

    @Test
    fun test_isinf_negative() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isinf_negative"))
    }

    @Test
    fun test_isinf_positive() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_isinf_positive"))
    }
}
