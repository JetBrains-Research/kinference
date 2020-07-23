package org.jetbrains.research.kotlin.inference.misc.pos

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class POSTest {
    private fun getTargetPath(dirName: String) = "/pos/$dirName/"

    @Test
    fun test_pos_tagger() {
        Utils.tensorTestRunner(getTargetPath("test_pos_tagger"))
    }
}
