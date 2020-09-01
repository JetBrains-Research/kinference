package io.kinference.operators.math

import io.kinference.Utils
import org.junit.jupiter.api.Test

class GeluTest {
    private fun getTargetPath(dirName: String) = "/gelu/$dirName/"

    @Test
    fun `test GELU`() {
        Utils.tensorTestRunner(getTargetPath("test_gelu"))
    }
}
