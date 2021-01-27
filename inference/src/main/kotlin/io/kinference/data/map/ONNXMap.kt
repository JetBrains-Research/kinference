package io.kinference.data.map

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.onnx.TensorProto
import io.kinference.types.ValueInfo
import io.kinference.types.ValueTypeInfo
import kotlin.collections.Map

class ONNXMap(val data: Map<Any, ONNXData>, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_MAP, info) {
    val keyType: TensorProto.DataType
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).keyType

    val valueType: ValueTypeInfo
        get() = (info.typeInfo as ValueTypeInfo.MapTypeInfo).valueType

    override fun rename(name: String): ONNXData = ONNXMap(data, ValueInfo(info.typeInfo, name))
}
