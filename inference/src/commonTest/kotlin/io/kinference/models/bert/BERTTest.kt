package io.kinference.models.bert

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class BERTTest {
    @Test
    fun heavy_test_vanilla_bert_model() = TestRunner.runTest {
        AccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    fun heavy_test_vanilla_bert_performance() = TestRunner.runTest {
        PerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
