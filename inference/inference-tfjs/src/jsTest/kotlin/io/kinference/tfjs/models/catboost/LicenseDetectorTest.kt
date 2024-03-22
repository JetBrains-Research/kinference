package io.kinference.tfjs.models.catboost

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.time.Duration

class LicenseDetectorTest {
    @Test
    fun heavy_test_license_detector() = runTest(timeout = Duration.INFINITE) {
        TFJSAccuracyRunner.runFromS3("catboost:license-detector:v1")
    }

    @Test
    fun benchmark_test_license_detector_performance() = runTest(timeout = Duration.INFINITE) {
        TFJSPerformanceRunner.runFromS3("catboost:license-detector:v1", count = 5)
    }
}
