package io.kinference.ort.models.gpt

import io.kinference.ort.runners.ORTTestEngine.ORTAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class ORTCommentUpdaterTest {
    @Test
    fun heavy_test_comment_updater() = TestRunner.runTest {
        ORTAccuracyRunner.runFromS3("custom:comment_updater")
    }
}
