package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class GECTest {
    @Test
    fun heavy_test_jvm_gec_model() = TestRunner.runTest(Platform.JVM) {
        KIAccuracyRunner.runFromS3(
            "bert:gec:en:standard:v2", disableTests = listOf(
                "test_data_set_batch_32_seqLen_32",
                "test_data_set_batch_32_seqLen_64",
                "test_data_set_batch_32_seqLen_92",
                "test_data_set_batch_32_seqLen_128",
                "test_data_set_batch_32_seqLen_256",
                "test_data_set_batch_32_seqLen_512",
            )
        )
    }

    @Test
    fun heavy_test_js_gec_model() = TestRunner.runTest(Platform.JS) {
        KIAccuracyRunner.runFromS3(
            "bert:gec:en:standard:v2", disableTests = listOf(
                "test_data_set_batch_32_seqLen_32",
                "test_data_set_batch_32_seqLen_64",
                "test_data_set_batch_32_seqLen_92",
                "test_data_set_batch_32_seqLen_128",
                "test_data_set_batch_32_seqLen_256",
                "test_data_set_batch_32_seqLen_512",
                "test_data_set_batch_1_seqLen_128",
                "test_data_set_batch_1_seqLen_256",
                "test_data_set_batch_1_seqLen_512"
            )
        )
    }

    @Test
    fun benchmark_test_gec_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:gec:en:standard:v2", count = 3)
    }
}
