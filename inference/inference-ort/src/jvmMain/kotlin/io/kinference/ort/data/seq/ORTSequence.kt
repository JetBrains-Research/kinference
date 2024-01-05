package io.kinference.ort.data.seq

import ai.onnxruntime.OnnxSequence
import io.kinference.data.ONNXDataType
import io.kinference.data.ONNXSequence
import io.kinference.ort.ORTBackend
import io.kinference.utils.ArrayUsageMarker

class ORTSequence(name: String?, override val data: OnnxSequence) : ONNXSequence<OnnxSequence, ORTBackend>(name, data) {
    override val backend: ORTBackend = ORTBackend

    override fun close() {
        data.close()
    }

    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    override fun rename(name: String): ORTSequence = ORTSequence(name, data)
    override fun markOutput(marker: ArrayUsageMarker) {
//        TODO("Not yet implemented")
    }

    override fun clone(newName: String?): ORTSequence {
        error("Cloning is not supported for OnnxRuntime backends")
    }
}
