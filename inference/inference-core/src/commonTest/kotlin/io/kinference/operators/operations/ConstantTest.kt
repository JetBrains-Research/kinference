package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ConstantTest {
    private fun getTargetPath(dirName: String) = "constant/$dirName/"

    @Test
    fun test_constant() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_constant"))
    }

    @Test
    fun test_scalar_constant() = runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_scalar_constant"))
    }
}
