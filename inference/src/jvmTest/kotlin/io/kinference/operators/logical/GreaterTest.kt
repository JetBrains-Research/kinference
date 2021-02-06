package io.kinference.operators.logical

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class GreaterTest {
    private fun getTargetPath(dirName: String) = "/greater/$dirName/"

    @Test
    fun `test greater`() {
        TestRunner.runFromResources(getTargetPath("test_greater"))
    }

    @Test
    fun `test greater with broadcast`() {
        TestRunner.runFromResources(getTargetPath("test_greater_bcast"))
    }
}
