package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test


class BERTTest {
    @Test
    fun heavy_test_vanilla_bert_model() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    fun benchmark_test_vanilla_bert_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
