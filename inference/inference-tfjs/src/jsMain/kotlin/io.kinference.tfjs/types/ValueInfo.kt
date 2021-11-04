package io.kinference.tfjs.types

import io.kinference.protobuf.message.ValueInfoProto

class ValueInfo(val typeInfo: ValueTypeInfo, val name: String = "") {
    companion object {
        fun create(proto: ValueInfoProto): ValueInfo {
            val type = proto.type!!
            val info = when {
                type.tensorType != null -> {
                    val dataType = type.tensorType!!.elem_type!!
                    val shape = if (type.tensorType?.shape == null) TensorShape.empty() else TensorShape(type.tensorType!!.shape!!)
                    ValueTypeInfo.TensorTypeInfo(shape, dataType)
                }
                type.sequenceType != null -> ValueTypeInfo.create(type)
                type.mapType != null -> ValueTypeInfo.create(type)
                else -> error("Unsupported data type")
            }
            return ValueInfo(info, proto.name ?: "")
        }
    }
}
