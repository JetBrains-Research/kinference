package io.kinference.webgpu.graph

import io.kinference.graph.Context
import io.kinference.graph.Graph
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.operators.WebGPUOperatorFactory

class WebGPUGraph(proto: GraphProto, opSetRegistry: OperatorSetRegistry) : Graph<WebGPUData<*>>(proto, opSetRegistry, WebGPUOperatorFactory) {
    override fun makeContext(root: Context<WebGPUData<*>>?): Context<WebGPUData<*>> =
        WebGPUContext((root as WebGPUContext).gpuState, root)

    override fun prepareInput(proto: TensorProto): WebGPUData<*> = WebGPUTensor.create(proto)
}
