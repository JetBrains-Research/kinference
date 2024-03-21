package io.kinference.models.catboost

import io.kinference.KITestEngine
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LicenseDetectorTest {
    @Test
    fun heavy_test_license_detector() = runTest {
        KITestEngine.KIAccuracyRunner.runFromS3("catboost:license-detector:v1")
    }

    @Test
    fun benchmark_test_license_detector_performance() = runTest {
        KITestEngine.KIPerformanceRunner.runFromS3("catboost:license-detector:v1", count = 5)
    }
}
