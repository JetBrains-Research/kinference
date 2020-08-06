package org.jetbrains.research.kotlin.inference.operators.math

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "/add/$dirName/"

    @Test
    fun `test add`() {
        Utils.tensorTestRunner(getTargetPath("test_add"))
    }

    @Test
    fun `test add broadcast`() {
        Utils.tensorTestRunner(getTargetPath("test_add_bcast"))
    }

    @Test
    fun `test add scalar`() {
        Utils.tensorTestRunner(getTargetPath("test_add_scalar"))
    }
}
