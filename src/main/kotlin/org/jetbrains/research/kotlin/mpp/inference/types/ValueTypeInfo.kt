package org.jetbrains.research.kotlin.mpp.inference.types

import TensorShapeProto
import TensorProto.DataType
import TypeProto

class TensorShape(val dims: IntArray) {

    //TODO: support params dimensions
    //class Dimension(val value: Int?, val param: String?)

    companion object {
        operator fun invoke(proto: TensorShapeProto): TensorShape {
            val dims = proto.dim.map { it.dim_value!!.toInt() }
            return TensorShape(dims.toIntArray())
        }
    }
}

abstract class ValueTypeInfo(val type: DataType) {
    companion object {
        fun create(proto: TypeProto) = when {
            proto.tensor_type != null -> TensorTypeInfo(proto.tensor_type)
            proto.sequence_type != null -> SequenceTypeInfo(proto.sequence_type)
            proto.map_type != null -> TODO("support maps")
            else -> error("One should be present")
        }
    }
}

class TensorTypeInfo(val shape: TensorShape, type: DataType) : ValueTypeInfo(type) {
    constructor(proto: TypeProto.Tensor) : this(TensorShape(proto.shape!!), DataType.fromValue(proto.elem_type!!)!!)
}

class SequenceTypeInfo(type: DataType) : ValueTypeInfo(type) {
    constructor(proto: TypeProto.Sequence) : this(DataType.fromValue(proto.elem_type!!.tensor_type!!.elem_type!!)!!)
}
