package io.kinference.models.catboost

import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class IJCompletionRankerTest {
    @Test
    fun heavy_test_ranker() = TestRunner.runTest {
        AccuracyRunner.runFromS3("catboost:ij-completion-ranker:v1")
    }

    @Test
    fun benchmark_test_ranker_performance() = TestRunner.runTest {
        PerformanceRunner.runFromS3("catboost:ij-completion-ranker:v1", count = 5)
    }
}
