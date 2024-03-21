package io.kinference.models

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class CommentUpdaterTest {
    @Test
    fun heavy_test_comment_updater() = runTest {
        KIAccuracyRunner.runFromS3("custom:comment_updater")
    }

    @Test
    fun benchmark_test_comment_updater() = runTest {
        KIPerformanceRunner.runFromS3("custom:comment_updater", count = 100)
    }
}
