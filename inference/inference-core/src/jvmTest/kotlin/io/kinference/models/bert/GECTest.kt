package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test


class GECTest {
    @Test
    fun heavy_test_gec_model() = TestRunner.runTest {
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
    fun benchmark_test_gec_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:gec:en:standard:v2", count = 3)
    }
}
