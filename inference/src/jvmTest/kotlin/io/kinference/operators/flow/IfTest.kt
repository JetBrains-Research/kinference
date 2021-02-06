package io.kinference.operators.flow

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class IfTest {
    private fun getTargetPath(dirName: String) = "/if/$dirName/"

    @Test
    fun `test if`() {
        TestRunner.runFromResources(getTargetPath("test_if"))
    }
}
