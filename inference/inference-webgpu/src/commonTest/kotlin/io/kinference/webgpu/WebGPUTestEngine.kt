package io.kinference.webgpu

import io.kinference.TestEngine
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.engine.WebGPUEngine
import io.kinference.webgpu.tensor.WebGPUTensor
import io.kinference.webgpu.utils.WebGPUAssertions
import kotlin.time.ExperimentalTime

object WebGPUTestEngine : TestEngine<WebGPUData<*>>(WebGPUEngine) {
    override fun checkEquals(expected: WebGPUData<*>, actual: WebGPUData<*>, delta: Double) {
        WebGPUAssertions.assertEquals(expected, actual, delta)
    }

    override fun postprocessData(data: WebGPUData<*>) {
        when (data) {
            is WebGPUTensor -> data.data.destroy()
        }
    }

    @OptIn(ExperimentalTime::class)
    val WebGPUAccuracyRunner = AccuracyRunner(WebGPUTestEngine)

    @OptIn(ExperimentalTime::class)
    val WebGPUPerformanceRunner = PerformanceRunner(WebGPUTestEngine)
}
