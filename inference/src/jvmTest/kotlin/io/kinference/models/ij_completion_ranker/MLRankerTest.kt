package io.kinference.models.ij_completion_ranker

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class MLRankerTest {
    @Test
    @Tag("heavy")
    fun `test ranker`()  = TestRunner.runTest {
        AccuracyRunner.runFromS3("catboost:ij-completion-ranker:v1")
    }

    @Test
    @Tag("heavy")
    fun `test ranker performance`()  = TestRunner.runTest {
        PerformanceRunner.runFromS3("catboost:ij-completion-ranker:v1", count = 5)
    }
}
