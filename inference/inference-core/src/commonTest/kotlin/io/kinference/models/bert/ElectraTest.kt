package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class ElectraTest {
    @Test
    fun heavy_test_electra() = runTest {
        KIAccuracyRunner.runFromS3("bert:electra")
    }

    @Test
    fun benchmark_test_electra() = runTest {
        KIPerformanceRunner.runFromS3("bert:electra", count = 5)
    }
}
