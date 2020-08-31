package io.kinference.operators.flow

import io.kinference.Utils
import org.junit.jupiter.api.Test

class LoopTest {
    private fun getTargetPath(dirName: String) = "/loop/$dirName/"

    @Test
    fun `test loop`() {
        Utils.tensorTestRunner(getTargetPath("test_loop"))
    }
}
