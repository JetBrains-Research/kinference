package io.kinference.ort.data

import ai.onnxruntime.*
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType

abstract class ORTData(override val data: OnnxValue, override val name: String?) : ONNXData<OnnxValue> {
    companion object {
        inline operator fun <reified V : OnnxValue> invoke(name: String?, data: V) : ORTData = when (data) {
            is OnnxTensor -> ORTTensor(data, name)
            is OnnxMap -> ORTMap(data, name)
            is OnnxSequence -> ORTSequence(data, name)
            else -> error("")
        }
    }
}

class ORTTensor(override val data: OnnxTensor, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
    override fun rename(name: String): ORTData = ORTTensor(data, name)
}

class ORTSequence(override val data: OnnxSequence, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    override fun rename(name: String): ORTData = ORTSequence(data, name)
}

class ORTMap(override val data: OnnxMap, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
    override fun rename(name: String): ORTData = ORTMap(data, name)
}

