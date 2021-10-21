package io.kinference.ort.data.seq

import ai.onnxruntime.OnnxSequence
import io.kinference.data.ONNXDataType
import io.kinference.data.ONNXSequence

class ORTSequence(name: String?, override val data: OnnxSequence) : ONNXSequence<OnnxSequence>(name, data) {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    override fun rename(name: String): ORTSequence = ORTSequence(name, data)
}
