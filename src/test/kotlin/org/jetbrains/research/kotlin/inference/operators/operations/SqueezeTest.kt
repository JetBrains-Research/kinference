package org.jetbrains.research.kotlin.inference.operators.operations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class SqueezeTest {
    private fun getTargetPath(dirName: String) = "/squeeze/$dirName/"

    @Test
    fun test_squeeze() {
        Utils.tensorTestRunner(getTargetPath("test_squeeze"))
    }

    @Test
    fun test_squeeze_negative_axes() {
        Utils.tensorTestRunner(getTargetPath("test_squeeze_negative_axes"))
    }
}
