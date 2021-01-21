package io.kinference.data.map

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.types.ValueInfo
import kotlin.collections.Map

class ONNXMap(val data: Map<Any, ONNXData>, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_MAP, info) {
    override fun rename(name: String): ONNXData = ONNXMap(data, ValueInfo(info.typeInfo, name))
}
