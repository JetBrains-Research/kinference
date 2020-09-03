package io.kinference.misc.pos

import io.kinference.Utils
import io.kinference.misc.ModelTestUtils
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class POSTest {
    private fun getTargetPath(dirName: String) = "/pos/$dirName/"

    @Test
    fun `test POS-tagger`() {
        Utils.tensorTestRunner(getTargetPath("test_pos_tagger"))
    }

    @Test
    @Tag("heavy")
    fun `test POS-tagger performance`() {
        ModelTestUtils.testModelPerformance("/pos/test_pos_tagger/")
    }
}
