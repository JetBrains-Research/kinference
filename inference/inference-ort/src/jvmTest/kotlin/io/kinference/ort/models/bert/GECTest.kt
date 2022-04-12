package io.kinference.ort.models.bert

import io.kinference.ort.ORTTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class GECTest {
    @Test
    fun heavy_test_gec_model() = TestRunner.runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromS3(
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
    fun benchmark_test_gec_performance() = TestRunner.runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromS3("bert:gec:en:standard:v2", count = 3)
    }
}
