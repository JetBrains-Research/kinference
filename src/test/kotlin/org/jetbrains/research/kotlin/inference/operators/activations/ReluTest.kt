package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class ReluTest {
    private fun getTargetPath(dirName: String) = "/relu/$dirName/"

    @Test
    fun test_relu(){
        Utils.tensorTestRunner(getTargetPath("test_relu"))
    }
}
