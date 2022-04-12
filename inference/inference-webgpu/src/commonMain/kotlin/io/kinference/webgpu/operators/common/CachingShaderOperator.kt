package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.NDArrayInfo

class CachedShaderOperatorInfo(
    val inputInfo: List<NDArrayInfo?>,
    val implementation: Operator<WebGPUTensor, WebGPUTensor>
)

abstract class CachingShaderOperator(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<WebGPUTensor, WebGPUTensor>(info, attributes, inputs, outputs) {
    private var cachedInfo: CachedShaderOperatorInfo? = null

    abstract fun operatorImplementation(inputInfo: List<NDArrayInfo?>, context: WebGPUContext): Operator<WebGPUTensor, WebGPUTensor>

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        val context = contexts.graph as WebGPUContext

        val inputInfo = inputs.map { it?.data?.info }
        if (inputInfo != cachedInfo?.inputInfo) {
            cachedInfo = CachedShaderOperatorInfo(
                inputInfo,
                operatorImplementation(inputInfo, context)
            )
        }
        return cachedInfo!!.implementation.apply(contexts, inputs)
    }
}

