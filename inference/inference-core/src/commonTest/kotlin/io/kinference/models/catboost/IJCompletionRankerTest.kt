package io.kinference.models.catboost

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class IJCompletionRankerTest {
    @Test
    fun heavy_test_ranker() = runTest {
        KIAccuracyRunner.runFromS3("catboost:ij-completion-ranker:v1")
    }

    @Test
    fun benchmark_test_ranker_performance() = runTest {
        KIPerformanceRunner.runFromS3("catboost:ij-completion-ranker:v1", count = 5)
    }
}
