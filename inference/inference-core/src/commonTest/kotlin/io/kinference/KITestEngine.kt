package io.kinference

import io.kinference.core.KIEngine
import io.kinference.data.ONNXData
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.KIAssertions
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
object KITestEngine : TestEngine(KIEngine) {
    override fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double) {
        KIAssertions.assertEquals(expected, actual, delta)
    }

    val KIAccuracyRunner = AccuracyRunner(KITestEngine)
    val KIPerformanceRunner = PerformanceRunner(KITestEngine)
}
