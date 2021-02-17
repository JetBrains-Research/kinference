package io.kinference.types

import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.protobuf.message.ValueInfoProto

class ValueInfo(val typeInfo: ValueTypeInfo, val name: String = "") {
    companion object {
        fun create(proto: ValueInfoProto): ValueInfo {
            val type = proto.type!!
            val info = when {
                type.tensor_type != null -> {
                    val dataType = DataType.fromValue(type.tensor_type!!.elem_type!!)!!
                    val shape = if (type.tensor_type?.shape == null) TensorShape.empty() else TensorShape(type.tensor_type!!.shape!!)
                    ValueTypeInfo.TensorTypeInfo(shape, dataType)
                }
                type.sequence_type != null -> ValueTypeInfo.create(type)
                type.map_type != null -> ValueTypeInfo.create(type)
                else -> error("Unsupported data type")
            }
            return ValueInfo(info, proto.name ?: "")
        }
    }
}
