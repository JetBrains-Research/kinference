package io.kinference.models.bert

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class GECTest {
    @Test
    @Tag("heavy")
    fun `test gec model`() {
        TestRunner.runFromS3("bert:gec:en:standard:v2")
    }

    @Test
    @Tag("heavy")
    fun `test gec performance`() {
        PerformanceRunner.runFromS3("bert:gec:en:standard:v2", count = 3)
    }
}
