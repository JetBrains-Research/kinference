package io.kinference.models

import io.kinference.runners.KITestEngine.KIAccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class CommentUpdaterTest {
    @Test
    fun heavy_test_comment_updater() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("custom:comment_updater")
    }

    @Test
    fun benchmark_test_comment_updater() = TestRunner.runTest {
        PerformanceRunner.runFromS3("custom:comment_updater", count = 100)
    }
}
