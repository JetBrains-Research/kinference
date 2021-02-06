package io.kinference.models.bert

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class BertTest {
    @Test
    @Tag("heavy")
    fun `test vanilla BERT model`() {
        TestRunner.runFromS3("/bert/v1/", "tests/bert/standard/en/v1/")
    }

    @Test
    @Tag("heavy")
    fun `test BERT performance`() {
        PerformanceRunner.runFromS3("/bert/v1/", "tests/bert/standard/en/v1/", count = 3)
    }
}
