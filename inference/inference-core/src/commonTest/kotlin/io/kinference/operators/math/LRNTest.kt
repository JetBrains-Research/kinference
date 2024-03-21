package io.kinference.operators.math

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LRNTest {
    private fun getTargetPath(dirName: String) = "LRN/$dirName/"

    @Test
    fun test_lrn() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_lrn"))
    }

    @Test
    fun test_lrn_default() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_lrn_default"))
    }
}
