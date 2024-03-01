package io.kinference.models

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ElectraTest {
    @Test
    fun heavy_test_electra() = TestRunner.runTest {
        KIAccuracyRunner.runFromS3("electra")
    }

    @Test
    fun benchmark_test_electra() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("electra", count = 5)
    }
}
