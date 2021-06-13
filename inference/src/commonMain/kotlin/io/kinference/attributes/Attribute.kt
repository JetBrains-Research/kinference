package io.kinference.attributes

import io.kinference.data.tensors.Tensor
import io.kinference.graph.Graph
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.AttributeProto.AttributeType
import kotlin.time.ExperimentalTime

class Attribute<T>(proto: AttributeProto, val value: T) {
    val name: String = proto.name!!
    val type: AttributeType = proto.type

    val refAttrName: String? = proto.refAttrName

    companion object {
        @ExperimentalTime
        fun create(proto: AttributeProto): Attribute<Any> = when (proto.type) {
            AttributeType.FLOAT -> Attribute(proto, proto.f!!)
            AttributeType.INT -> Attribute(proto, proto.i!!)
            AttributeType.STRING -> Attribute(proto, proto.s!!)
            AttributeType.TENSOR -> Attribute(proto, Tensor.create(proto.t!!))
            AttributeType.GRAPH -> Attribute(proto, Graph(proto.g!!))
            AttributeType.SPARSE_TENSOR -> TODO("Not supported in current version of KInference")
            AttributeType.FLOATS -> Attribute(proto, proto.floats!!)
            AttributeType.INTS -> Attribute(proto, proto.ints!!)
            AttributeType.STRINGS -> Attribute(proto, proto.strings)
            AttributeType.TENSORS -> Attribute(proto, proto.tensors.map { Tensor.create(it) })
            AttributeType.GRAPHS -> Attribute(proto, proto.graphs.map { Graph(it) })
            AttributeType.SPARSE_TENSORS -> TODO("Not supported in current version of KInference")
            AttributeType.UNDEFINED -> error("Cannot get attribute ${proto.name} type")
        }
    }
}
