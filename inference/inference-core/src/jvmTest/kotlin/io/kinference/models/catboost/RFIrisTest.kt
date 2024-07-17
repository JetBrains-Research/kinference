package io.kinference.models.catboost

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class RFIrisTest {
    @Test
    fun heavy_test_rf_iris() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("rf_iris/")
    }
}
