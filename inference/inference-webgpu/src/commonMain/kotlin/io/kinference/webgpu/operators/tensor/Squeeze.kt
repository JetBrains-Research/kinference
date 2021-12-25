package io.kinference.webgpu.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.toIntArray
import io.kinference.operator.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.*

sealed class Squeeze(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<WebGPUTensor, WebGPUTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in SqueezeVer1.VERSION.asRange() -> SqueezeVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Squeeze operator: $version")
        }
    }
}

@ExperimentalTime
class SqueezeVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Squeeze(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "squeezed", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("Squeeze", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axes: LongArray? by attributeOrNull()

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<WebGPUTensor?>, profilingContext: ProfilingContext?): List<WebGPUTensor?> {
        context as WebGPUContext
        val squeezeAxes = axes?.toIntArray() ?: IntArray(0)

        return listOf(inputs[0]!!.data.squeeze(squeezeAxes, context.gpuState).asTensor())
    }
}
