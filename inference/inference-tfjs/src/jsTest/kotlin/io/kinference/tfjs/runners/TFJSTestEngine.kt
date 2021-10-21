package io.kinference.tfjs.runners

import io.kinference.TestEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.tfjs.TFJSEngine
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.utils.TFJSAssertions

object TFJSTestEngine : TestEngine(TFJSEngine) {
    override fun checkEquals(expected: ONNXData<*>, actual: ONNXData<*>, delta: Double) {
        TFJSAssertions.assertEquals(expected, actual, delta)
    }

    override fun postprocessData(data: ONNXData<*>) {
        when (data.type) {
            ONNXDataType.ONNX_TENSOR -> (data as TFJSTensor).data.dispose()
            ONNXDataType.ONNX_SEQUENCE -> (data as TFJSSequence).data.forEach { postprocessData(it) }
            ONNXDataType.ONNX_MAP -> (data as TFJSMap).data.values.forEach { postprocessData(it) }
        }
    }

    val TFJSAccuracyRunner = AccuracyRunner(TFJSTestEngine)
    val TFJSPerformanceRunner = PerformanceRunner(TFJSTestEngine)
}
