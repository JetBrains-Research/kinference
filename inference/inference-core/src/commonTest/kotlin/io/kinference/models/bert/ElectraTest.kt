package io.kinference.models.bert

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlin.test.Test

class ElectraTest {
    @Test
    fun heavy_test_jvm_electra() = TestRunner.runTest(Platform.JVM) {
        KIAccuracyRunner.runFromS3("bert:electra")
    }

    @Test
    fun benchmark_test_electra_performance() = TestRunner.runTest {
        KIPerformanceRunner.runFromS3("bert:electra", count = 5)
    }
}
