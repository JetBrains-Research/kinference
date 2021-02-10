package io.kinference.models.bert

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class BertTest {
    @Test
    @Tag("heavy")
    fun `test vanilla BERT model`() = TestRunner.runTest {
        AccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    @Tag("heavy")
    fun `test BERT performance`() = TestRunner.runTest {
        PerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
