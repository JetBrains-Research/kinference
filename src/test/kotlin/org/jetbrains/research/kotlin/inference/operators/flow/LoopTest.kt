package org.jetbrains.research.kotlin.inference.operators.flow

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class LoopTest {
    private fun getTargetPath(dirName: String) = "/loop/$dirName/"

    @Test
    fun `test loop`() {
        Utils.tensorTestRunner(getTargetPath("test_loop"))
    }
}
