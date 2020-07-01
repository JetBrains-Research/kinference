package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class TanhTest {
    private fun getTargetPath(dirName: String) = "/tanh/$dirName/"

    @Test
    fun test_tanh_example(){
        Utils.tensorTestRunner(getTargetPath("test_tanh_example"))
    }

    @Test
    fun test_tanh(){
        Utils.tensorTestRunner(getTargetPath("test_tanh"))
    }
}
