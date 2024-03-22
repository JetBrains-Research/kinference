package io.kinference.operators.math

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LRNTest {
    private fun getTargetPath(dirName: String) = "LRN/$dirName/"

    @Test
    fun test_lrn() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_lrn"))
    }

    @Test
    fun test_lrn_default() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_lrn_default"))
    }
}
