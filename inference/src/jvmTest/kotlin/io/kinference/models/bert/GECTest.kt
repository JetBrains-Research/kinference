package io.kinference.models.bert

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class GECTest {
    @Test
    @Tag("heavy")
    fun `test gec model`() {
        TestRunner.runFromS3("/tests/bert/gec/en/standard/v1/", "tests/bert/gec/en/standard/v1/")
    }

    @Test
    @Tag("heavy")
    fun `test gec performance`() {
        PerformanceRunner.runFromS3("/tests/bert/gec/en/standard/v1/", "tests/bert/gec/en/standard/v1/", count = 3)
    }
}
