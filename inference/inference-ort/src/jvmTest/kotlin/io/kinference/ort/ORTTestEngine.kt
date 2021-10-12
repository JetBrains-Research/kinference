package io.kinference.ort

import io.kinference.TestEngine
import io.kinference.data.ONNXData
import io.kinference.ort.utils.ORTAssertions
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import kotlin.time.ExperimentalTime

object ORTTestEngine : TestEngine(ORTEngine) {
    override fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double) {
        ORTAssertions.assertEquals(expected, actual, delta)
    }

    @OptIn(ExperimentalTime::class)
    val ORTAccuracyRunner = AccuracyRunner(ORTTestEngine)

    @OptIn(ExperimentalTime::class)
    val ORTPerformanceRunner = PerformanceRunner(ORTTestEngine)
}
