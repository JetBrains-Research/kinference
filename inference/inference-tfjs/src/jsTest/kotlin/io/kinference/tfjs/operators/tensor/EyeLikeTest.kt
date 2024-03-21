package io.kinference.tfjs.operators.tensor

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class EyeLikeTest {
    private fun getTargetPath(dirName: String) = "eyelike/$dirName/"

    @Test
    fun test_eyelike_with_dtype() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_eyelike_with_dtype"))
    }

    @Test
    fun test_eyelike_populate_off_main_diagonal() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_eyelike_populate_off_main_diagonal"))
    }

    @Test
    fun test_eyelike_without_dtype() = runTest {
        TFJSAccuracyRunner.runFromResources(getTargetPath("test_eyelike_without_dtype"))
    }
}
