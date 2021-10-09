package io.kinference.ort.data.map

import ai.onnxruntime.OnnxMap
import io.kinference.data.ONNXDataType
import io.kinference.ort.data.ORTData

class ORTMap(override val data: OnnxMap, name: String?) : ORTData(data, name) {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
    override fun rename(name: String): ORTData = ORTMap(data, name)
}
