package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Unsqueeze(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in UnsqueezeVer1.VERSION.asRange() -> UnsqueezeVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Constant operator: $version")
            }
    }
}

class UnsqueezeVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Unsqueeze(name, INFO, attributes, inputs, outputs) {
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

    private val axes: Array<Int> by attribute { array: LongArray -> Array(array.size) { array[it].toInt() } }

    private fun Int.indexAxis(limitAxis: Int) = if (this < 0) this + limitAxis else this

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        require(axes.size == axes.toSet().size) { "Axes must contains only unique elements, present: ${axes.joinToString(prefix = "[", postfix = "]")}" }

        val output = tidyNDArray {
            val input = inputs[0]!!.data
            val actualAxes = axes.map { it.indexAxis(input.rank + axes.size) }.sorted()
            val newShape = input.shape.toMutableList()
            for (axis in actualAxes) {
                newShape.add(axis, 1)
            }
            return@tidyNDArray input.reshape(newShape.toIntArray())
        }
        return listOf(output.asTensor("expanded"))
    }
}
