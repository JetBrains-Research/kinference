package io.kinference.models

import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class POSTest {
    @Test
    fun heavy_test_pos_tagger()  = TestRunner.runTest {
        AccuracyRunner.runFromResources("/pos_tagger/")
    }

    @Test
    fun benchmark_test_pos_tagger_performance()  = TestRunner.runTest {
        PerformanceRunner.runFromResources("/pos_tagger/")
    }
}
