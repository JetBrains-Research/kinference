package io.kinference.tfjs.models.catboost

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.tfjs.runners.TFJSTestEngine.TFJSPerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class LicenseDetectorTest {
    @Test
    fun heavy_test_license_detector() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromS3("catboost:license-detector:v1")
    }

    @Test
    fun benchmark_test_license_detector_performance() = TestRunner.runTest {
        TFJSPerformanceRunner.runFromS3("catboost:license-detector:v1", count = 5)
    }
}
