package io.kinference.ort

import ai.onnxruntime.OnnxValue
import io.kinference.TestEngine
import io.kinference.ort.utils.ORTAssertions
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import kotlin.time.ExperimentalTime

object ORTTestEngine : TestEngine<ORTData<*>>(ORTEngine) {
    override fun checkEquals(expected: ORTData<*>, actual: ORTData<*>, delta: Double) {
        ORTAssertions.assertEquals(expected, actual, delta)
    }

    override fun postprocessData(data: ORTData<*>) {
        val onnxData = data.data
        if (onnxData is OnnxValue) {
            onnxData.close()
        }
    }

    @OptIn(ExperimentalTime::class)
    val ORTAccuracyRunner = AccuracyRunner(ORTTestEngine)

    @OptIn(ExperimentalTime::class)
    val ORTPerformanceRunner = PerformanceRunner(ORTTestEngine)
}
