package io.kinference.ort

import ai.onnxruntime.OnnxValue
import io.kinference.TestEngine
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.utils.ORTAssertions
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import kotlin.time.ExperimentalTime

object ORTTestEngine : TestEngine<ORTData<*>>(ORTEngine) {
    override fun checkEquals(expected: ORTData<*>, actual: ORTData<*>, delta: Double) {
        ORTAssertions.assertEquals(expected, actual, delta)
    }

    @OptIn(ExperimentalTime::class)
    val ORTAccuracyRunner = AccuracyRunner(ORTTestEngine)

    @OptIn(ExperimentalTime::class)
    val ORTPerformanceRunner = PerformanceRunner(ORTTestEngine)
}
