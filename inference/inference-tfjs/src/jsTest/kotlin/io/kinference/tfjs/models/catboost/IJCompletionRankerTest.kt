package io.kinference.tfjs.models.catboost

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration


class IJCompletionRankerTest {
    @Test
    fun heavy_test_ranker() = runTest(timeout = Duration.INFINITE) {
        TFJSAccuracyRunner.runFromS3("catboost:ij-completion-ranker:v1")
    }

    @Test
    fun benchmark_test_ranker_performance() = runTest(timeout = Duration.INFINITE) {
        TFJSPerformanceRunner.runFromS3("catboost:ij-completion-ranker:v1", count = 5)
    }
}
