package io.kinference.ort.models.bert

import io.kinference.ort.ORTTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test


class BERTTest {
    @Test
    fun heavy_test_vanilla_bert_model() = TestRunner.runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    fun benchmark_test_vanilla_bert_performance() = TestRunner.runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
