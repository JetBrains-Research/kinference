package io.kinference.webgpu.model

import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.webgpu.*
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.graph.WebGPUGraph
import io.kinference.webgpu.ndarray.NDArray
import io.kinference.webgpu.ndarray.asTensor
import io.kinference.webgpu.operators.WebGPUOperatorFactory
import io.kinference.webgpu.tensor.WebGPUTensor
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class WebGPUModel(proto: ModelProto, val device: Device) : Model<WebGPUData<*>> {
    val name: String = "${proto.domain}:${proto.modelVersion}"
    private val opSet = OperatorSetRegistry(proto.opSetImport)

    private val commandEncoder = device.createCommandEncoder()
    private val operatorFactory = WebGPUOperatorFactory(device, commandEncoder)
    private val graph = WebGPUGraph(proto.graph!!, opSet, device, commandEncoder, operatorFactory)

    override fun predict(input: List<WebGPUData<*>>, profile: Boolean): Map<String, WebGPUData<*>> {
        error("Use predictSuspend with WebGPU backend")
    }

    override suspend fun predictSuspend(input: List<WebGPUData<*>>, profile: Boolean): Map<String, WebGPUData<*>> {
        input.forEach {
            (it as? WebGPUTensor)?.data?.unmap()
        }
        val outputs = graph.execute(input).associateBy { it.name.orEmpty() }

        val outputTensors = outputs.mapValues { (name, tensor) ->
            val outputNDArrray = NDArray((tensor as WebGPUTensor).data.info, isOutput = true, device = device)
            commandEncoder.copyBufferToBuffer(tensor.data.buffer, 0, outputNDArrray.buffer, 0, tensor.data.info.sizeBytes)
            outputNDArrray.asTensor(name)
        }
        val commandBuffer = commandEncoder.finish()
        device.queue.submit(listOf(commandBuffer))
        outputTensors.values.forEach {
            it.data.map(MapModeFlags(MapMode.Read))
        }
        return outputTensors
    }
}
