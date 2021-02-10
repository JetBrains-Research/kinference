package io.kinference.models.ij_completion_ranker

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class MLRankerTest {
    @Test
    @Tag("heavy")
    fun `test ranker`() {
        TestRunner.runFromS3("catboost:ij-completion-ranker:v1")
    }

    @Test
    @Tag("heavy")
    fun `test ranker performance`() {
        PerformanceRunner.runFromS3("catboost:ij-completion-ranker:v1", count = 5)
    }
}
