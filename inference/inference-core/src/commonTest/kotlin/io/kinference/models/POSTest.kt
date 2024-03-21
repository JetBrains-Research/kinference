package io.kinference.models

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class POSTest {
    @Test
    fun heavy_test_pos_tagger() = runTest {
        KIAccuracyRunner.runFromResources("pos_tagger/")
    }

    @Test
    fun benchmark_test_pos_tagger_performance() = runTest {
        KIPerformanceRunner.runFromResources("pos_tagger/")
    }
}
