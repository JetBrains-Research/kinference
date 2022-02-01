package io.kinference.ort_gpu.data.map

import ai.onnxruntime.OnnxMap
import io.kinference.data.*
import io.kinference.ort_gpu.ORTGPUBackend

class ORTGPUMap(name: String?, override val data: OnnxMap) : ONNXMap<OnnxMap, ORTGPUBackend>(name, data) {
    override val backend: ORTGPUBackend = ORTGPUBackend

    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
    override fun rename(name: String): ORTGPUMap = ORTGPUMap(name, data)
}
