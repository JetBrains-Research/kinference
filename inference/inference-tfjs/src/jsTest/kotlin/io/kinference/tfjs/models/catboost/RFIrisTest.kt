package io.kinference.tfjs.models.catboost

import io.kinference.tfjs.runners.TFJSTestEngine.TFJSAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class RFIrisTest {
    @Test
    fun heavy_test_rf_iris() = TestRunner.runTest {
        TFJSAccuracyRunner.runFromResources("rf_iris/")
    }
}
