package io.kinference.ort

import io.kinference.TestEngine
import io.kinference.ort.utils.ORTAssertions
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner

object ORTTestEngine : TestEngine<ORTData<*>>(ORTEngine) {
    override fun checkEquals(expected: ORTData<*>, actual: ORTData<*>, delta: Double) {
        ORTAssertions.assertEquals(expected, actual, delta)
    }

        val ORTAccuracyRunner = AccuracyRunner(ORTTestEngine)

        val ORTPerformanceRunner = PerformanceRunner(ORTTestEngine)
}
