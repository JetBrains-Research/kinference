package io.kinference.ort.models

import io.kinference.ort.ORTTestEngine.ORTAccuracyRunner
import io.kinference.ort.ORTTestEngine.ORTPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test


class CommentUpdaterTest {
    @Test
    fun gpu_test_comment_updater() = runTest {
        ORTAccuracyRunner.runFromS3("custom:comment_updater")
    }

    @Test
    fun benchmark_test_comment_updater() = runTest {
        ORTPerformanceRunner.runFromS3("custom:comment_updater", count = 100)
    }
}
