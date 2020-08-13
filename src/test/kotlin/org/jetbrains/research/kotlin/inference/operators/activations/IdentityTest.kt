package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class IdentityTest {
    private fun getTargetPath(dirName: String) = "/identity/$dirName/"

    @Test
    fun `test identity`() {
        Utils.tensorTestRunner(getTargetPath("test_identity"))
    }
}
