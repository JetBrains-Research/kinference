package io.kinference.ort.data.map

import ai.onnxruntime.OnnxMap
import io.kinference.data.ONNXDataType
import io.kinference.data.ONNXMap
import io.kinference.ort.ORTBackend

class ORTMap(name: String?, override val data: OnnxMap) : ONNXMap<OnnxMap, ORTBackend>(name, data) {
    override val backend: ORTBackend = ORTBackend

    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
    override fun rename(name: String): ORTMap = ORTMap(name, data)

    override fun close() {
        data.close()
    }
}
