package io.kinference.models.bert

import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class GECTest {
    @Test
    @Tag("heavy")
    fun `test gec model`() = TestRunner.runTest {
        AccuracyRunner.runFromS3("bert:gec:en:standard:v2")
    }

    @Test
    @Tag("heavy")
    fun `test gec performance`() = TestRunner.runTest {
        PerformanceRunner.runFromS3("bert:gec:en:standard:v2", count = 3)
    }
}
