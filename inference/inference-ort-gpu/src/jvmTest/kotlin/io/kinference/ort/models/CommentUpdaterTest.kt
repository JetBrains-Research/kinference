package io.kinference.ort.models

import io.kinference.ort.ORTTestEngine.ORTAccuracyRunner
import io.kinference.ort.ORTTestEngine.ORTPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class CommentUpdaterTest {
    @Test
    fun gpu_test_comment_updater() = TestRunner.runTest {
        ORTAccuracyRunner.runFromS3("custom:comment_updater")
    }

    @Test
    fun benchmark_test_comment_updater() = TestRunner.runTest {
        ORTPerformanceRunner.runFromS3("custom:comment_updater", count = 100)
    }
}
