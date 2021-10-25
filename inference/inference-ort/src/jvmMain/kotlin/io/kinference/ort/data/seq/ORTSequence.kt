package io.kinference.ort.data.seq

import ai.onnxruntime.OnnxSequence
import io.kinference.data.*
import io.kinference.ort.ORTBackend

class ORTSequence(name: String?, override val data: OnnxSequence) : ONNXSequence<OnnxSequence, ORTBackend>(name, data) {
    override val backend: ORTBackend = ORTBackend

    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    override fun rename(name: String): ORTSequence = ORTSequence(name, data)
}
