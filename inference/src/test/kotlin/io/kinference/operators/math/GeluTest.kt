package io.kinference.operators.math

import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Test

class GeluTest {
    private fun getTargetPath(dirName: String) = "/gelu/$dirName/"

    @Test
    fun `test GELU`() {
        TestRunner.runFromResources(getTargetPath("test_gelu"))
    }
}
