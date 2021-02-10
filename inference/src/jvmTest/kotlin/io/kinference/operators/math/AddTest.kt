package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "/add/$dirName/"

    @Test
    fun `test add`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun `test add broadcast`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun `test add scalar`()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
