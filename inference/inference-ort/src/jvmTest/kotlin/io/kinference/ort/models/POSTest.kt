package io.kinference.ort.models

import io.kinference.ort.ORTTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class POSTest {
    @Test
    fun heavy_test_pos_tagger() = TestRunner.runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromResources("pos_tagger/")
    }

    @Test
    fun benchmark_test_pos_tagger_performance() = TestRunner.runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromResources("pos_tagger/")
    }
}
