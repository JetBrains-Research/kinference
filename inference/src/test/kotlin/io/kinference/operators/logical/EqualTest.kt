package io.kinference.operators.logical

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class EqualTest {
    private fun getTargetPath(dirName: String) = "/equal/$dirName/"

    @Test
    fun `test equal`() {
        TestRunner.runFromResources(getTargetPath("test_equal"))
    }

    @Test
    fun `test equal with broadcast`() {
        TestRunner.runFromResources(getTargetPath("test_equal_bcast"))
    }
}
