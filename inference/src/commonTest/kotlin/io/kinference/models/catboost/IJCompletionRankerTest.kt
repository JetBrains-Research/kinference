package io.kinference.models.catboost

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class IJCompletionRankerTest {
    @Test
    fun heavy_test_ranker() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("catboost:ij-completion-ranker:v1")
    }

    @Test
    fun benchmark_test_ranker_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("catboost:ij-completion-ranker:v1", count = 5)
    }
}
