package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration

class GECTest {
    @Test
    fun heavy_test_gec_model() = runTest(timeout = Duration.INFINITE) {
        val disabledTests = when (PlatformUtils.platform) {
            Platform.JVM -> listOf(
                "test_data_set_batch_32_seqLen_32",
                "test_data_set_batch_32_seqLen_64",
                "test_data_set_batch_32_seqLen_92",
                "test_data_set_batch_32_seqLen_128",
                "test_data_set_batch_32_seqLen_256",
                "test_data_set_batch_32_seqLen_512",
            )
            Platform.JS -> listOf(
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
            else -> error(platformNotSupportedMessage)
        }
        KIAccuracyRunner.runFromS3("bert:gec:en:standard:v2", disableTests = disabledTests)
    }

    @Test
    fun benchmark_test_gec_performance() = runTest(timeout = Duration.INFINITE) {
        KIPerformanceRunner.runFromS3("bert:gec:en:standard:v2", count = 3)
    }
}
