package io.kinference.webgpu.operators

import io.kinference.attribute.Attribute
import io.kinference.attribute.AttributeFactory
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.tensor.WebGPUTensor
import io.kinference.graph.Graph
import io.kinference.operator.*
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.webgpu.CommandEncoder
import io.kinference.utils.webgpu.Device
import io.kinference.webgpu.graph.WebGPUGraph
import io.kinference.webgpu.operators.math.*
import io.kinference.webgpu.operators.tensor.*
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class WebGPUOperatorFactory(device: Device, commandEncoder: CommandEncoder) : OperatorFactory<WebGPUData<*>> {
    inner class WebGPUAttributeFactory(private val device: Device, private val commandEncoder: CommandEncoder) : AttributeFactory<WebGPUData<*>> {
        override fun createTensor(proto: TensorProto): WebGPUData<*> = WebGPUTensor.create(proto, device)
        override fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<WebGPUData<*>> =
            WebGPUGraph(proto, opSet, device = device, commandEncoder = commandEncoder, this@WebGPUOperatorFactory)
    }

    private val attributeFactory = WebGPUAttributeFactory(device, commandEncoder)

    override fun attributeFactory() = attributeFactory

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
        "Div" -> Div(version, attributes, inputs, outputs)
        "Flatten" -> Flatten(version, attributes, inputs, outputs)
        "MatMul" -> MatMul(version, attributes, inputs, outputs)
        "Mul" -> Mul(version, attributes, inputs, outputs)
        "Reshape" -> Reshape(version, attributes, inputs, outputs)
        "Sub" -> Sub(version, attributes, inputs, outputs)
        /*

         */
        else -> error("Unsupported operator: $opType")
    } as Operator<WebGPUData<*>, WebGPUData<*>>
}
