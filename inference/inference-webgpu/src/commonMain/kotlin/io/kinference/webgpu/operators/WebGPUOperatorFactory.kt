package io.kinference.webgpu.operators

import io.kinference.attribute.Attribute
import io.kinference.attribute.AttributeFactory
import io.kinference.graph.Graph
import io.kinference.operator.*
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.graph.WebGPUGraph
import io.kinference.webgpu.operators.logical.*
import io.kinference.webgpu.operators.math.*
import io.kinference.webgpu.operators.tensor.*

object WebGPUAttributeFactory : AttributeFactory<WebGPUData<*>> {
    override fun createTensor(proto: TensorProto): WebGPUData<*> = WebGPUTensor.create(proto)
    override fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<WebGPUData<*>> = WebGPUGraph(proto, opSet)
}

object WebGPUOperatorFactory : OperatorFactory<WebGPUData<*>> {
    override fun attributeFactory() = WebGPUAttributeFactory

    @Suppress("UNCHECKED_CAST")
    override fun create(
        opType: String?,
        version: Int?,
        attributes: Map<String, Attribute<Any>>,
        inputs: List<String>,
        outputs: List<String>
    ): Operator<WebGPUData<*>, WebGPUData<*>>  = when (opType) {
        "Add" -> Add(version, attributes, inputs, outputs)
        "Constant" -> Constant(version, attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(version, attributes, inputs, outputs)
        "Div" -> Div(version, attributes, inputs, outputs)
        "Equal" -> Equal(version, attributes, inputs, outputs)
        "Flatten" -> Flatten(version, attributes, inputs, outputs)
        "Greater" -> Greater(version, attributes, inputs, outputs)
        "Less" -> Less(version, attributes, inputs, outputs)
        "MatMul" -> MatMul(version, attributes, inputs, outputs)
        "Mul" -> Mul(version, attributes, inputs, outputs)
        "Or" -> Or(version, attributes, inputs, outputs)
        "Reshape" -> Reshape(version, attributes, inputs, outputs)
        "Squeeze" -> Squeeze(version, attributes, inputs, outputs)
        "Sub" -> Sub(version, attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(version, attributes, inputs, outputs)
        else -> error("Unsupported operator: $opType")
    } as Operator<WebGPUData<*>, WebGPUData<*>>
}
