package io.kinference.operators.operations

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class EyeLikeTest {
    private fun getTargetPath(dirName: String) = "eyelike/$dirName/"

    @Test
    fun test_eyelike_with_dtype() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_eyelike_with_dtype"))
    }

    @Test
    fun test_eyelike_populate_off_main_diagonal() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_eyelike_populate_off_main_diagonal"))
    }

    @Test
    fun test_eyelike_without_dtype() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromResources(getTargetPath("test_eyelike_without_dtype"))
    }
}
