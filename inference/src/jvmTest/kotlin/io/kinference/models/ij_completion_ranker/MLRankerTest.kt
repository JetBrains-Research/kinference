package io.kinference.models.ij_completion_ranker

import io.kinference.runners.PerformanceRunner
import io.kinference.runners.TestRunner
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test

class MLRankerTest {
    @Test
    @Tag("heavy")
    fun `test ranker`() {
        TestRunner.runFromS3(TEST_PATH, PREFIX)
    }

    @Test
    @Tag("heavy")
    fun `test ranker performance`() {
        PerformanceRunner.runFromS3(TEST_PATH, PREFIX)
    }

    companion object {
        const val TEST_PATH = "/catboost/ij_completion_ranker/"
        const val PREFIX = "tests/catboost/ij_completion_ranker/"
    }
}
