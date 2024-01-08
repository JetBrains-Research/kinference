package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test


class BERTTest {
    @Test
    fun heavy_test_jvm_vanilla_bert_model() = TestRunner.runTest(Platform.JVM) {
        KIAccuracyRunner.runFromS3("bert:standard:en:v1")
    }

    @Test
    fun heavy_test_js_vanilla_bert_model() = TestRunner.runTest(Platform.JS) {
        KIAccuracyRunner.runFromS3("bert:standard:en:v1", disableTests = listOf(
            "test_data_set_batch8_seq40"
        ))
    }

    @Test
    fun benchmark_test_vanilla_bert_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
