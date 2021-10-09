package io.kinference.ort.runners

import io.kinference.TestEngine
import io.kinference.data.ONNXData
import io.kinference.ort.ORTEngine
import io.kinference.ort.utils.ORTAssertions
import io.kinference.runners.AccuracyRunner
import kotlin.time.ExperimentalTime

object ORTTestEngine : TestEngine(ORTEngine) {
    override fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double) {
        ORTAssertions.assertEquals(expected, actual, delta)
    }

    @OptIn(ExperimentalTime::class)
    val ORTAccuracyRunner = AccuracyRunner(ORTTestEngine)
}
