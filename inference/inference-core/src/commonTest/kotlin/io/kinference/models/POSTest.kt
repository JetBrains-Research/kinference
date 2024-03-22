package io.kinference.models

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration


class POSTest {
    @Test
    fun heavy_test_pos_tagger() = runTest(timeout = Duration.INFINITE) {
        KIAccuracyRunner.runFromResources("pos_tagger/")
    }

    @Test
    fun benchmark_test_pos_tagger_performance() = runTest(timeout = Duration.INFINITE) {
        KIPerformanceRunner.runFromResources("pos_tagger/")
    }
}
