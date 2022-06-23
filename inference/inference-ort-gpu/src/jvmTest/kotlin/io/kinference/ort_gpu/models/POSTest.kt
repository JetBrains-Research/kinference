package io.kinference.ort_gpu.models

import io.kinference.ort_gpu.ORTGPUTestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class POSTest {
    @Test
    fun heavy_test_pos_tagger() = TestRunner.runTest {
        ORTGPUTestEngine.ORTGPUAccuracyRunner.runFromResources("pos_tagger/")
    }

    @Test
    fun benchmark_test_pos_tagger_performance() = TestRunner.runTest {
        ORTGPUTestEngine.ORTGPUPerformanceRunner.runFromResources("pos_tagger/")
    }
}
