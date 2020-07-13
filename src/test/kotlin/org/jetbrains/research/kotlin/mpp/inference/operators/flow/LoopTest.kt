package org.jetbrains.research.kotlin.mpp.inference.operators.flow

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class LoopTest {
    private fun getTargetPath(dirName: String) = "/loop/$dirName/"

    @Test
    fun test_loop() {
        Utils.tensorTestRunner(getTargetPath("test_loop"))
    }
}
