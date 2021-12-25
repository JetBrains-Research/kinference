package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.NDArrayInfo
import io.kinference.utils.webgpu.*
import io.kinference.graph.Context
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.profiler.ProfilingContext

class CachedOperatorInfo(
    val inputInfo: List<NDArrayInfo?>,
    val outputInfo: List<NDArrayInfo?>,
    val bindGroupLayout: BindGroupLayout,
    val computePipeline: ComputePipeline,
    val workGroupSize: IntArray,
    val dispatchSize: IntArray
)

abstract class WebGPUTensorOperator(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<WebGPUTensor, WebGPUTensor>(info, attributes, inputs, outputs) {
    private var cachedOperatorInfo: CachedOperatorInfo? = null

    protected abstract val shaderEntryPoint: String

    protected abstract fun outputInfo(inputInfo: List<NDArrayInfo?>): List<NDArrayInfo?>

    protected abstract fun workGroupSize(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>): IntArray
    protected abstract fun dispatchSize(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>, workGroupSize: IntArray): IntArray

    protected abstract fun createBindGroupLayout(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>): BindGroupLayoutDescriptor

    protected abstract fun createShader(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>): String

    private fun createComputePipeline(context: WebGPUContext, inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>): ComputePipeline {
        val bindGroupLayout = context.gpuState.device.createBindGroupLayout(createBindGroupLayout(inputInfo, outputInfo))
        val pipelineLayout = context.gpuState.device.createPipelineLayout(
            PipelineLayoutDescriptor(bindGroupLayouts = listOf(bindGroupLayout))
        )
        return context.gpuState.device.createComputePipeline(
            ComputePipelineDescriptor(
                layout = pipelineLayout,
                compute = ProgrammableStage(
                    module = context.gpuState.device.createShaderModule(ShaderModuleDescriptor(createShader(inputInfo, outputInfo))),
                    entryPoint = shaderEntryPoint
                )
            )
        )
    }

    protected abstract fun apply(context: WebGPUContext, inputs: List<WebGPUTensor?>, operatorInfo: CachedOperatorInfo): List<WebGPUTensor?>

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<WebGPUTensor?>, profilingContext: ProfilingContext?): List<WebGPUTensor?> {
        context as WebGPUContext
        val inputInfo = inputs.map { it?.data?.info }
        if (inputInfo != cachedOperatorInfo?.inputInfo) {
            val outputInfo = outputInfo(inputInfo)
            val workGroupSize = workGroupSize(inputInfo, outputInfo)

            cachedOperatorInfo = CachedOperatorInfo(
                inputInfo = inputInfo,
                outputInfo = outputInfo(inputInfo),
                bindGroupLayout = context.gpuState.device.createBindGroupLayout(createBindGroupLayout(inputInfo, outputInfo)),
                computePipeline = createComputePipeline(context as WebGPUContext, inputInfo, outputInfo),
                workGroupSize = workGroupSize,
                dispatchSize = dispatchSize(inputInfo, outputInfo, workGroupSize),
            )
        }
        return apply(context as WebGPUContext, inputs, cachedOperatorInfo!!)
    }
}
