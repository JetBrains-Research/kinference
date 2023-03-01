package io.kinference.attribute

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.*
import io.kinference.utils.Closeable
import kotlin.time.ExperimentalTime

class Attribute<T>(proto: AttributeProto, val value: T): Closeable {
    val name: String = proto.name!!
    val type: AttributeProto.AttributeType = proto.type

    val refAttrName: String? = proto.refAttrName

    override fun close() {
        if (value is Closeable) value.close()

        if (value is List<*>) {
            for (element in value) {
                if (element is Closeable) element.close()
            }
        }
    }
}

interface AttributeFactory<T : ONNXData<*, *>> {
    fun createTensor(proto: TensorProto): T
    fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<T>

    @ExperimentalTime
    fun create(proto: AttributeProto, currentOpSet: OperatorSetRegistry): Attribute<Any> = when (proto.type) {
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
