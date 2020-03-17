package org.jetbrains.research.kotlin.mpp.inference.attributes

import AttributeProto
import AttributeProto.*
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class Attribute<T>(proto: AttributeProto, val value: T) {
    val name: String = proto.name!!
    val refAttrName: String? = proto.ref_attr_name

    companion object {
        fun create(proto: AttributeProto): Attribute<*> = when(proto.type) {
            AttributeType.FLOAT -> Attribute(proto, proto.f!!)
            AttributeType.INT -> Attribute(proto, proto.i!!)
            AttributeType.STRING -> Attribute(proto, proto.s!!)
            AttributeType.TENSOR -> Attribute(proto, Tensor.create(proto.t!!))
            AttributeType.GRAPH -> TODO()
            AttributeType.SPARSE_TENSOR -> TODO()
            AttributeType.FLOATS -> Attribute(proto, proto.floats)
            AttributeType.INTS -> Attribute(proto, proto.ints)
            AttributeType.STRINGS -> Attribute(proto, proto.strings)
            AttributeType.TENSORS -> Attribute(proto, proto.tensors.map { Tensor.create(it) })
            AttributeType.GRAPHS -> TODO()
            AttributeType.SPARSE_TENSORS -> TODO()
            else -> throw IllegalStateException("Unsupported attribute type")
        }
    }
}
