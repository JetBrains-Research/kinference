package io.kinference.types

import io.kinference.onnx.TensorProto.DataType
import io.kinference.onnx.ValueInfoProto

sealed class ValueInfo(val name: String) {
    companion object {
        fun create(proto: ValueInfoProto): ValueInfo {
            val type = proto.type!!
            return when {
                type.tensor_type != null -> {
                    val dataType = DataType.fromValue(type.tensor_type.elem_type!!)!!
                    val shape = if (type.tensor_type.shape == null) TensorShape.empty() else TensorShape(type.tensor_type.shape)
                    TensorInfo(proto.name!!, dataType, shape)
                }
                type.sequence_type != null -> {
                    val elementTypes = ValueTypeInfo.create(type.sequence_type.elem_type!!)
                    SequenceInfo(proto.name!!, elementTypes)
                }
                type.map_type != null -> {
                    val keyType = DataType.fromValue(type.map_type.key_type!!)!!
                    val valueType = ValueTypeInfo.create(type.map_type.value_type!!)
                    MapInfo(proto.name!!, keyType, valueType)
                }
                else -> error("Unsupported data type")
            }
        }
    }

    class TensorInfo(name: String, val type: DataType, val shape: TensorShape) : ValueInfo(name)

    class SequenceInfo(name: String, val type: ValueTypeInfo) : ValueInfo(name)

    class MapInfo(name: String, val keyType: DataType, val valueType: ValueTypeInfo) : ValueInfo(name)
}
