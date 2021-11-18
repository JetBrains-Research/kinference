package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.ndarray.toIntArray
import io.kinference.operator.*
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.extensions.*

sealed class Unsqueeze(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<TFJSTensor, TFJSTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in UnsqueezeVer1.VERSION.asRange() -> UnsqueezeVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

class UnsqueezeVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Unsqueeze(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "expanded", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("Unsqueeze", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axes: IntArray by attribute { it: LongArray -> it.toIntArray() }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<TFJSTensor?>, profilingContext: ProfilingContext?): List<TFJSTensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val actualAxes = axes.map { input.indexAxis(it) }.sorted()
            val newShape = input.shape.toMutableList()
            for (axis in actualAxes) {
                newShape.add(axis, 1)
            }
            return@tidy arrayOf(input.reshape(newShape.toTypedArray()))
        }
        return listOf(outputs[0].asTensor("expanded"))
    }
}
