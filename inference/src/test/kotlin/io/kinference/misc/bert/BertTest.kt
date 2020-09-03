package io.kinference.misc.bert

import io.kinference.Utils
import io.kinference.misc.ModelTestUtils
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class BertTest {
    private fun getTargetPath(dirName: String) = "/bert/$dirName/"

    @Test
    @Tag("heavy")
    fun `test vanilla BERT model`() {
        Utils.tensorTestRunner(getTargetPath("v1"))
    }

    @Test
    @Tag("heavy")
    fun `test BERT performance`() {
        ModelTestUtils.testModelPerformance("/bert/v1/")
    }
}
