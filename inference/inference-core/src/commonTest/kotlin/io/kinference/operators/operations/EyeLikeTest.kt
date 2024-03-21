package io.kinference.operators.operations

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class EyeLikeTest {
    private fun getTargetPath(dirName: String) = "eyelike/$dirName/"

    @Test
    fun test_eyelike_with_dtype() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_eyelike_with_dtype"))
    }

    @Test
    fun test_eyelike_populate_off_main_diagonal() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_eyelike_populate_off_main_diagonal"))
    }

    @Test
    fun test_eyelike_without_dtype() = runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_eyelike_without_dtype"))
    }
}
