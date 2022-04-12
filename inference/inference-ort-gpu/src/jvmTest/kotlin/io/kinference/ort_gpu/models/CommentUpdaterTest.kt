package io.kinference.ort_gpu.models

import io.kinference.ort_gpu.ORTGPUTestEngine.ORTGPUAccuracyRunner
import io.kinference.ort_gpu.ORTGPUTestEngine.ORTGPUPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class CommentUpdaterTest {
    @Test
    fun heavy_test_comment_updater() = TestRunner.runTest {
        ORTGPUAccuracyRunner.runFromS3("custom:comment_updater")
    }

    @Test
    fun benchmark_test_comment_updater() = TestRunner.runTest {
        ORTGPUPerformanceRunner.runFromS3("custom:comment_updater", count = 100)
    }
}
