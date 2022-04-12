package io.kinference.ort_gpu

import ai.onnxruntime.OnnxValue
import io.kinference.TestEngine
import io.kinference.ort_gpu.data.map.ORTGPUMap
import io.kinference.ort_gpu.data.seq.ORTGPUSequence
import io.kinference.ort_gpu.data.tensor.ORTGPUTensor
import io.kinference.ort_gpu.utils.ORTGPUAssertions
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import kotlin.time.ExperimentalTime

object ORTGPUTestEngine : TestEngine<ORTGPUData<*>>(ORTGPUEngine) {
    override fun checkEquals(expected: ORTGPUData<*>, actual: ORTGPUData<*>, delta: Double) {
        ORTGPUAssertions.assertEquals(expected, actual, delta)
    }

    override fun postprocessData(data: ORTGPUData<*>) {
        val onnxData = data.data
        if (onnxData is OnnxValue) {
            onnxData.close()
        }
    }

    @OptIn(ExperimentalTime::class)
    val ORTGPUAccuracyRunner = AccuracyRunner(ORTGPUTestEngine)

    @OptIn(ExperimentalTime::class)
    val ORTGPUPerformanceRunner = PerformanceRunner(ORTGPUTestEngine)
}
