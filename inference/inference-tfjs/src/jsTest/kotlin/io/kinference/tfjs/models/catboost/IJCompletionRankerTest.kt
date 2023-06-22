package io.kinference.tfjs.models.catboost

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test


class IJCompletionRankerTest {
    @Test
    fun heavy_test_ranker() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromS3("catboost:ij-completion-ranker:v1")
    }

    @Test
    fun benchmark_test_ranker_performance() = TestRunner.runTest {
        TFJSPerformanceRunner.runFromS3("catboost:ij-completion-ranker:v1", count = 5)
    }
}
