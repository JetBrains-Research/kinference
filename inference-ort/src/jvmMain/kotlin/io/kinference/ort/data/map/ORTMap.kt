package io.kinference.ort.data.map

import ai.onnxruntime.OnnxMap
import io.kinference.data.ONNXDataType
import io.kinference.data.ONNXMap

class ORTMap(name: String?, override val data: OnnxMap) : ONNXMap<OnnxMap>(name, data) {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
    override fun rename(name: String): ORTMap = ORTMap(name, data)
}
