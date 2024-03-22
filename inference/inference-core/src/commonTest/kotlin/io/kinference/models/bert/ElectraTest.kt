package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration

class ElectraTest {
    @Test
    fun heavy_test_electra() = runTest(timeout = Duration.INFINITE) {
        KIAccuracyRunner.runFromS3("bert:electra")
    }

    @Test
    fun benchmark_test_electra() = runTest(timeout = Duration.INFINITE) {
        KIPerformanceRunner.runFromS3("bert:electra", count = 5)
    }
}
