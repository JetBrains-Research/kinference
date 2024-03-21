package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ElectraTest {
    @Test
    fun heavy_test_electra() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("bert:electra")
    }

    @Test
    fun benchmark_test_electra() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:electra", count = 5)
    }
}
