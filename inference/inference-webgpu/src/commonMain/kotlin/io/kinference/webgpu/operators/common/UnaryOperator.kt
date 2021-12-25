package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.operator.OperatorInfo
import io.kinference.utils.webgpu.*
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.*

abstract class UnaryOperator(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : WebGPUTensorOperator(info, attributes, inputs, outputs) {
    override val shaderEntryPoint: String = "main"

    override fun createBindGroupLayout(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>): BindGroupLayoutDescriptor = BindGroupLayoutDescriptor(
        listOf(
            BindGroupLayoutEntry(0, BufferBindingLayout(BufferBindingType.ReadOnlyStorage)),
            BindGroupLayoutEntry(1, BufferBindingLayout(BufferBindingType.Storage))
        )
    )

    override fun apply(context: WebGPUContext, inputs: List<WebGPUTensor?>, operatorInfo: CachedOperatorInfo): List<WebGPUTensor?> {
        val outputInfo = operatorInfo.outputInfo[0]!!
        val output = NDArray(outputInfo, context.gpuState)
        val bindGroup = context.gpuState.device.createBindGroup(
            BindGroupDescriptor(
                layout = operatorInfo.bindGroupLayout,
                entries = listOf(
                    BindGroupEntry(0, BufferBinding(inputs[0]!!.data.getBuffer(context.gpuState))),
                    BindGroupEntry(1, BufferBinding(output.getBuffer(context.gpuState))),
                )
            )
        )

        val computePass = context.gpuState.beginComputePass()
        computePass.setPipeline(operatorInfo.computePipeline)
        computePass.setBindGroup(0, bindGroup, listOf())
        computePass.dispatch(
            operatorInfo.dispatchSize[0],
            operatorInfo.dispatchSize[1],
            operatorInfo.dispatchSize[2],
        )
        computePass.endPass()
        return listOf(output.asTensor("C"))
    }
}
