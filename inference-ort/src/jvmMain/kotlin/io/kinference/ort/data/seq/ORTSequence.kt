package io.kinference.ort.data.seq

import ai.onnxruntime.OnnxSequence
import io.kinference.data.ONNXDataType
import io.kinference.ort.data.ORTData

class ORTSequence(override val data: OnnxSequence, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    override fun rename(name: String): ORTData = ORTSequence(data, name)
}
