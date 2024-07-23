package io.kinference.models

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.KITestEngine.KIPerformanceRunner
import io.kinference.utils.Platform
import io.kinference.utils.TestRunner
import kotlinx.coroutines.async
import kotlin.test.Test

class SimultaneousModelsTest {
    @Test
    fun heavy_test_simultaneous_bert_electra_models() = TestRunner.runTest {
        val bertDeferred = async {
            KIAccuracyRunner.runFromS3("bert:standard:en:v1", errorsVerbose = false)
        }

        val electraDeferred = async {
            KIAccuracyRunner.runFromS3("bert:electra", errorsVerbose = false)
        }

        bertDeferred.await()
        electraDeferred.await()
    }

    @Test
    fun benchmark_test_simultaneous_bert_electra_performance() = TestRunner.runTest {
        val bertDeferred = async {
            KIPerformanceRunner.runFromS3("bert:standard:en:v1", count = 50)
        }

        val electraDeferred = async {
            KIPerformanceRunner.runFromS3("bert:electra", count = 5)
        }

        bertDeferred.await()
        electraDeferred.await()
    }
}
