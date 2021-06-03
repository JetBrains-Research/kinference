package io.kinference.models.bert

import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class GECTest {
//    @Test
    fun heavy_test_gec_model() = TestRunner.runTest {
        AccuracyRunner.runFromS3("bert:gec:en:standard:v2")
    }

    @Test
    fun benchmark_test_gec_performance() = TestRunner.runTest {
        PerformanceRunner.runFromS3("bert:gec:en:standard:v2", count = 3)
    }
}
