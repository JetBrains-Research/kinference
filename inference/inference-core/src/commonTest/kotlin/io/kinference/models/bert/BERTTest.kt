package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration

class BERTTest {
    @Test
    fun heavy_test_vanilla_bert_model() = runTest(timeout = Duration.INFINITE) {
        val disabledTests = when (PlatformUtils.platform) {
            Platform.JVM -> listOf()
            Platform.JS -> listOf(
                "test_data_set_batch8_seq40"
            )
            else -> error(platformNotSupportedMessage)
        }
        KIAccuracyRunner.runFromS3("bert:standard:en:v1", disableTests = disabledTests)
    }

    @Test
    fun benchmark_test_vanilla_bert_performance() = runTest(timeout = Duration.INFINITE) {
        KIPerformanceRunner.runFromS3("bert:standard:en:v1", count = 3)
    }
}
