package io.kinference.attribute

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.*
import kotlin.time.ExperimentalTime

class Attribute<T>(proto: AttributeProto, val value: T) {
    val name: String = proto.name!!
    val type: AttributeProto.AttributeType = proto.type

    val refAttrName: String? = proto.refAttrName
}

interface AttributeFactory<T : ONNXData<*, *>> {
    fun createTensor(proto: TensorProto): T
    suspend fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<T>

    @ExperimentalTime
    suspend fun create(proto: AttributeProto, currentOpSet: OperatorSetRegistry): Attribute<Any> = when (proto.type) {
        AttributeProto.AttributeType.FLOAT -> Attribute(proto, proto.f!!)
        AttributeProto.AttributeType.INT -> Attribute(proto, proto.i!!)
        AttributeProto.AttributeType.STRING -> Attribute(proto, proto.s!!)
        AttributeProto.AttributeType.TENSOR -> Attribute(proto, createTensor(proto.t!!))
        AttributeProto.AttributeType.GRAPH -> Attribute(proto, createGraph(proto.g!!, currentOpSet))
        AttributeProto.AttributeType.SPARSE_TENSOR -> TODO("Not supported in the current version of KInference")
        AttributeProto.AttributeType.FLOATS -> Attribute(proto, proto.floats!!)
        AttributeProto.AttributeType.INTS -> Attribute(proto, proto.ints!!)
        AttributeProto.AttributeType.STRINGS -> Attribute(proto, proto.strings)
        AttributeProto.AttributeType.TENSORS -> Attribute(proto, proto.tensors.map { createTensor(it) })
        AttributeProto.AttributeType.GRAPHS -> Attribute(proto, proto.graphs.map { createGraph(it, currentOpSet) })
        AttributeProto.AttributeType.SPARSE_TENSORS -> TODO("Not supported in the current version of KInference")
        AttributeProto.AttributeType.UNDEFINED -> error("Cannot get attribute ${proto.name} type")
    }
}
