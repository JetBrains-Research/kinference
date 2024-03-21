package io.kinference.ort.models.bert

import io.kinference.ort.ORTTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class BERTTest {
    @Test
    fun heavy_test_vanilla_bert_model() = runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    fun benchmark_test_vanilla_bert_performance() = runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
