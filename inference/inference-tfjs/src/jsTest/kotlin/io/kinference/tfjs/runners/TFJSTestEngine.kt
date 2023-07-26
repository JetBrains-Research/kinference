package io.kinference.tfjs.runners

import io.kinference.MemoryProfileable
import io.kinference.TestEngine
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.core.memory
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.TFJSEngine
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.utils.TFJSAssertions

object TFJSTestEngine : TestEngine<TFJSData<*>>(TFJSEngine), MemoryProfileable {
    override fun checkEquals(expected: TFJSData<*>, actual: TFJSData<*>, delta: Double) {
        TFJSAssertions.assertEquals(expected, actual, delta)
    }

    override fun getInMemorySize(data: TFJSData<*>): Int {
        return when(data.type) {
            ONNXDataType.ONNX_TENSOR -> 1
            ONNXDataType.ONNX_SEQUENCE -> (data as TFJSSequence).data.sumOf { getInMemorySize(it) }
            ONNXDataType.ONNX_MAP -> (data as TFJSMap).data.values.sumOf { getInMemorySize(it) }
        }
    }

    override fun allocatedMemory(): Int = memory().numTensors

    val TFJSAccuracyRunner = AccuracyRunner(TFJSTestEngine)
    val TFJSPerformanceRunner = PerformanceRunner(TFJSTestEngine)
}
