package io.kinference.operators.activations

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class ErfTest {
    private fun getTargetPath(dirName: String) = "/erf/$dirName/"

    @Test
    fun `test erf`() {
        TestRunner.runFromResources(getTargetPath("test_erf"))
    }
}
