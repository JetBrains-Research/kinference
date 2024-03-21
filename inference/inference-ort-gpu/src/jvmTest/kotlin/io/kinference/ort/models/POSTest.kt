package io.kinference.ort.models

import io.kinference.ort.ORTTestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class POSTest {
    @Test
    fun gpu_test_pos_tagger() = runTest {
        ORTTestEngine.ORTAccuracyRunner.runFromResources("pos_tagger/")
    }

    @Test
    fun benchmark_test_pos_tagger_performance() = runTest {
        ORTTestEngine.ORTPerformanceRunner.runFromResources("pos_tagger/")
    }
}
