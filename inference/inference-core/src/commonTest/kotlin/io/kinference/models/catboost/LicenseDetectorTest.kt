package io.kinference.models.catboost

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration

class LicenseDetectorTest {
    @Test
    fun heavy_test_license_detector() = runTest(timeout = Duration.INFINITE) {
        KITestEngine.KIAccuracyRunner.runFromS3("catboost:license-detector:v1")
    }

    @Test
    fun benchmark_test_license_detector_performance() = runTest(timeout = Duration.INFINITE) {
        KITestEngine.KIPerformanceRunner.runFromS3("catboost:license-detector:v1", count = 5)
    }
}
