package io.kinference.ort

import ai.onnxruntime.*
import io.kinference.TestEngine
import io.kinference.data.ONNXDataType
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.utils.ORTAssertions
import io.kinference.ort.utils.ORTErrors
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.Errors

object ORTTestEngine : TestEngine<ORTData<*>>(ORTEngine) {
    override fun checkEquals(expected: ORTData<*>, actual: ORTData<*>, delta: Double) {
        ORTAssertions.assertEquals(expected, actual, delta)
    }

    override fun calculateErrors(expected: ORTData<*>, actual: ORTData<*>): List<Errors.ErrorsData> {
        return ORTErrors.calculateErrors(expected, actual)
    }

    private fun getInMemorySize(data: OnnxValue): Int {
        return when(data) {
            is OnnxTensor -> 1
            is OnnxSequence -> data.value.sumOf { getInMemorySize(it) }
            is OnnxMap -> data.size()
            else -> error("Unsupported data type")
        }
    }

    override fun getInMemorySize(data: ORTData<*>): Int {
        return when(data.type) {
            ONNXDataType.ONNX_TENSOR -> 1
            ONNXDataType.ONNX_SEQUENCE -> (data as ORTSequence).data.value.sumOf { getInMemorySize(it) }
            ONNXDataType.ONNX_MAP -> getInMemorySize((data as ORTMap).data)
        }
    }

    val ORTAccuracyRunner = AccuracyRunner(ORTTestEngine)

    val ORTPerformanceRunner = PerformanceRunner(ORTTestEngine)
}
