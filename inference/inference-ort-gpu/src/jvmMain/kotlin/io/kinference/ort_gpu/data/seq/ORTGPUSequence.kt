package io.kinference.ort_gpu.data.seq

import ai.onnxruntime.OnnxSequence
import io.kinference.data.*
import io.kinference.ort_gpu.ORTGPUBackend

class ORTGPUSequence(name: String?, override val data: OnnxSequence) : ONNXSequence<OnnxSequence, ORTGPUBackend>(name, data) {
    override val backend: ORTGPUBackend = ORTGPUBackend

    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
    override fun rename(name: String): ORTGPUSequence = ORTGPUSequence(name, data)
}
