package io.kinference.operator

import io.kinference.attribute.Attribute
import io.kinference.attribute.AttributeFactory
import io.kinference.data.ONNXData
import io.kinference.protobuf.message.NodeProto

interface OperatorFactory<T : ONNXData<*, *>> {
    fun attributeFactory(): AttributeFactory<T>
    fun create(name: String, opType: String?, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Operator<T, T>

    suspend fun create(proto: NodeProto, opSetRegistry: OperatorSetRegistry): Operator<T, T> {
        val version = opSetRegistry.getVersion(proto.domain)
        val attributes = proto.attribute.map { attributeFactory().create(it, opSetRegistry) }.associateBy { it.name }
        return create(proto.name ?: "", proto.opType, version, attributes, proto.input, proto.output)
    }
}
