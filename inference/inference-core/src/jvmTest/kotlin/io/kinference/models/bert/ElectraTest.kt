package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ElectraTest {
    @Test
    fun heavy_test_electra() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("bert:electra", errorsVerbose = false)
    }

    @Test
    fun benchmark_test_electra_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:electra", count = 5)
    }

    @Test
    fun benchmark_test_electra_coroutines() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:electra", count = 5, warmup = 0, parallelLoad = true)
    }
}
