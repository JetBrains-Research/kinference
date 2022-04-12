package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.OperatorInfo
import io.kinference.profiler.ProfilingContext
import io.kinference.utils.webgpu.*
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.*

abstract class UnaryOperator(
    device: Device,
    info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : ShaderOperator(device, info, attributes, inputs, outputs) {
    override val bindGroupLayoutDescriptor: BindGroupLayoutDescriptor = BindGroupLayoutDescriptor(
        listOf(
            BindGroupLayoutEntry(0, BufferBindingLayout(BufferBindingType.ReadOnlyStorage)),
            BindGroupLayoutEntry(1, BufferBindingLayout(BufferBindingType.Storage))
        )
    )

    abstract val outputShape: IntArray
    abstract val outputType: WebGPUDataType

    protected val outputInfo: NDArrayInfo
        get() = NDArrayInfo(outputShape, outputType)

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        val context = contexts.graph as WebGPUContext

        val output = NDArray(outputInfo, context.gpuState)
        val bindGroup = context.gpuState.device.createBindGroup(
            BindGroupDescriptor(
                layout = device.createBindGroupLayout(bindGroupLayoutDescriptor),
                entries = listOf(
                    BindGroupEntry(0, BufferBinding(inputs[0]!!.data.getBuffer(context.gpuState))),
                    BindGroupEntry(1, BufferBinding(output.getBuffer(context.gpuState))),
                )
            )
        )
        performComputePass(context.gpuState, bindGroup)
        return listOf(output.asTensor("output"))
    }
}
