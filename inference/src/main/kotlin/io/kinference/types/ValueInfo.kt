package io.kinference.types

import io.kinference.onnx.TensorProto.DataType
import io.kinference.onnx.ValueInfoProto

//TODO: optionally support maps and sequences
abstract class ValueInfo(val name: String, val type: DataType) {
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
                    val tensorTypes = type.sequence_type.elem_type!!.tensor_type!!
                    SequenceInfo(proto.name!!, DataType.fromValue(tensorTypes.elem_type!!)!!)
                }
                type.map_type != null -> TODO("Maps are not supported")
                else -> error("Unsupported data type")
            }
        }
    }
}

class TensorInfo(name: String, type: DataType, val shape: TensorShape) : ValueInfo(name, type)

class SequenceInfo(name: String, type: DataType) : ValueInfo(name, type)
