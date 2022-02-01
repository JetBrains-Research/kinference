package io.kinference.ort_gpu.models.bert

import io.kinference.ort_gpu.ORTGPUTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class BERTTest {
    @Test
    fun heavy_test_vanilla_bert_model() = TestRunner.runTest {
        ORTGPUTestEngine.ORTGPUAccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    fun benchmark_test_vanilla_bert_performance() = TestRunner.runTest {
        ORTGPUTestEngine.ORTGPUPerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
