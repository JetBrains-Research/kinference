package org.jetbrains.research.kotlin.mpp.inference.operators.math

import org.jetbrains.research.kotlin.mpp.inference.Utils
import org.junit.jupiter.api.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "/add/$dirName/"

    @Test
    fun test_add(){
        Utils.tensorTestRunner(getTargetPath("test_add"))
    }

    @Test
    fun test_add_bcast(){
        Utils.tensorTestRunner(getTargetPath("test_add_bcast"))
    }
}
