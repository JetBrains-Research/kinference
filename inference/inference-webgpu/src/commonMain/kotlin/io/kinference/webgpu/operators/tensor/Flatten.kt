package io.kinference.webgpu.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.operator.*
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.*
import io.kinference.webgpu.tensor.WebGPUTensor

sealed class Flatten(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<WebGPUTensor, WebGPUTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 5, untilVersion = 14)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in FlattenVer1.VERSION.asRange() -> FlattenVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

class FlattenVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Flatten(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 5, untilVersion = 14)
        private val INFO = OperatorInfo("Flatten", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<WebGPUTensor?>, profilingContext: ProfilingContext?): List<WebGPUTensor?> {
        context as WebGPUContext

        val data = inputs[0]!!.data.getMappedRange()
        val type = inputs[0]!!.data.info.type
        val targetShape = intArrayOf(inputs[0]!!.data.info.shape.size)
        return listOf(NDArray(ArrayInfo(shape = targetShape, type = type), data = data, device = context.device).asTensor())
    }
}
