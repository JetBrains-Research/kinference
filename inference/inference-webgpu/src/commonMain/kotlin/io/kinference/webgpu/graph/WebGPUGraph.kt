package io.kinference.webgpu.graph

import io.kinference.graph.Context
import io.kinference.graph.Graph
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.webgpu.CommandEncoder
import io.kinference.utils.webgpu.Device
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.operators.WebGPUOperatorFactory
import io.kinference.webgpu.tensor.WebGPUTensor
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class WebGPUGraph(
    proto: GraphProto, opSetRegistry: OperatorSetRegistry,
    private val device: Device,
    private val commandEncoder: CommandEncoder,
    operatorFactory: WebGPUOperatorFactory,
) : Graph<WebGPUData<*>>(proto, opSetRegistry, operatorFactory) {

    override fun makeContext(root: Context<WebGPUData<*>>?): Context<WebGPUData<*>> = WebGPUContext(device, commandEncoder, root as? WebGPUContext)
    override fun prepareInput(proto: TensorProto): WebGPUData<*> = WebGPUTensor.create(proto, device)
}
