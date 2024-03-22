package io.kinference.models

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration

class CommentUpdaterTest {
    @Test
    fun heavy_test_comment_updater() = runTest(timeout = Duration.INFINITE) {
        KIAccuracyRunner.runFromS3("custom:comment_updater")
    }

    @Test
    fun benchmark_test_comment_updater() = runTest(timeout = Duration.INFINITE) {
        KIPerformanceRunner.runFromS3("custom:comment_updater", count = 100)
    }
}
