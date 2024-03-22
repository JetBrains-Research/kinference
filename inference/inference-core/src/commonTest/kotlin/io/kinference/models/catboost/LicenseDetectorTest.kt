package io.kinference.models.catboost

import io.kinference.KITestEngine
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LicenseDetectorTest {
    @Test
    fun heavy_test_license_detector() = TestRunner.runTest {
        KITestEngine.KIAccuracyRunner.runFromS3("catboost:license-detector:v1")
    }

    @Test
    fun benchmark_test_license_detector_performance() = TestRunner.runTest {
        KITestEngine.KIPerformanceRunner.runFromS3("catboost:license-detector:v1", count = 5)
    }
}
