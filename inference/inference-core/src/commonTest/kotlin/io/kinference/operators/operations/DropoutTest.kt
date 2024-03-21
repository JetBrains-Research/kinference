package io.kinference.operators.operations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class DropoutTest {
    private fun getTargetPath(dirName: String) = "dropout/$dirName/"

    @Test
    fun test_dropout_default() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default"))
    }

    @Test
    fun test_dropout_default_mask() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default_mask"))
    }

    @Test
    fun test_dropout_default_mask_ratio() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default_mask_ratio"))
    }

    @Test
    fun test_dropout_default_ratio() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default_ratio"))
    }
}
