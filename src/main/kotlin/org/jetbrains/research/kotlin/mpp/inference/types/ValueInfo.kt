package org.jetbrains.research.kotlin.mpp.inference.types

import TensorProto
import TensorShapeProto
import ValueInfoProto

//TODO: optionally support maps and sequences
class ValueInfo(val name: String, val shape: TensorShape, val type: TensorProto.DataType) {
    companion object {
        fun create(proto: ValueInfoProto): ValueInfo {
            val tensor = proto.type?.tensor_type
            requireNotNull(tensor) { "Only tensor types are supported" }

            val shape = TensorShape.invoke(tensor.shape!!)
            val typeInt = tensor.elem_type!!

            return ValueInfo(proto.name!!, shape, TensorProto.DataType.fromValue(typeInt)!!)
        }
    }
}

class TensorShape(val dims: List<Dimension>) {
    class Dimension(values: Long?, params: String?)

    companion object {
        operator fun invoke(proto: TensorShapeProto): TensorShape {
            val dims = proto.dim.map { Dimension(it.dim_value, it.dim_param) }
            return TensorShape(dims)
        }
    }
}
