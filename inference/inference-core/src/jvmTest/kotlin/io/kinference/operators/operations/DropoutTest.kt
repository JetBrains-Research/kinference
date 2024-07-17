package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class DropoutTest {
    private fun getTargetPath(dirName: String) = "dropout/$dirName/"

    @Test
    fun test_dropout_default() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default"))
    }

    @Test
    fun test_dropout_default_mask() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default_mask"))
    }

    @Test
    fun test_dropout_default_mask_ratio() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default_mask_ratio"))
    }

    @Test
    fun test_dropout_default_ratio() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_dropout_default_ratio"))
    }
}
