package io.kinference.webgpu.model

import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.ModelProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.engine.WebGPUEnvironment
import io.kinference.webgpu.graph.*

class WebGPUModel(proto: ModelProto) : Model<WebGPUData<*>> {
    val name: String = "${proto.domain}:${proto.modelVersion}"
    private val opSet = OperatorSetRegistry(proto.opSetImport)
    private val graph = WebGPUGraph(proto.graph!!, opSet)

    override fun predict(input: List<WebGPUData<*>>, profile: Boolean): Map<String, WebGPUData<*>> {
        error("Use predictSuspend with WebGPU backend")
    }

    override suspend fun predictSuspend(input: List<WebGPUData<*>>, profile: Boolean): Map<String, WebGPUData<*>> {
        val context = WebGPUContext(WebGPUState(WebGPUEnvironment.getDevice()))
        val outputs = graph.executeSuspend(input, context).associateBy { it.name.orEmpty() }
        outputs.forEach { (_, value) ->
            if (value is WebGPUTensor) {
                value.data.requestData(context.gpuState)
            }
        }
        outputs.forEach { (_, value) ->
            if (value is WebGPUTensor) {
                value.data.finalizeOutputNDArray(context.gpuState)
            }
        }
        context.destroyRemovedValues()
        return outputs
    }
}
