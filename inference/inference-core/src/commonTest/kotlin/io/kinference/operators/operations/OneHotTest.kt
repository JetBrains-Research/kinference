package io.kinference.operators.operations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class OneHotTest {
    private fun getTargetPath(dirName: String) = "onehot/$dirName/"

    @Test
    fun test_onehot_with_axis() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_onehot_with_axis"))
    }

    @Test
    fun test_onehot_without_axis() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_onehot_without_axis"))
    }

    @Test
    fun test_onehot_with_negative_axis() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_onehot_with_negative_axis"))
    }

    @Test
    fun test_onehot_negative_indices() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_onehot_negative_indices"))
    }
}
