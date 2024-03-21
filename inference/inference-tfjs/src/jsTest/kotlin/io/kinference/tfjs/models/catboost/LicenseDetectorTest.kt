package io.kinference.tfjs.models.catboost

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSPerformanceRunner
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

class LicenseDetectorTest {
    @Test
    fun heavy_test_license_detector() = runTest {
        TFJSAccuracyRunner.runFromS3("catboost:license-detector:v1")
    }

    @Test
    fun benchmark_test_license_detector_performance() = runTest {
        TFJSPerformanceRunner.runFromS3("catboost:license-detector:v1", count = 5)
    }
}
