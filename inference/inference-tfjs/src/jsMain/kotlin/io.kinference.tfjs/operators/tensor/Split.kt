package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.indexAxis
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.toIntArray
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Split(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 2, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Split {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SplitVer2.VERSION.asRange() -> SplitVer2(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Constant operator: $version")
            }
        }
    }
}


class SplitVer2(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Split(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L),
            AttributeInfo("split", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false, differentiable = true))

        private val OUTPUTS_INFO = listOf(VariadicIOInfo(0, TYPE_CONSTRAINTS, "outputs", minimumArity = 1, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 2, untilVersion = 13)
        private val INFO = OperatorInfo("Split", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val split: LongArray? by attributeOrNull()

    @Suppress("UNCHECKED_CAST")
    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val actualAxis = input.indexAxis(axis)
        return if (split == null) {
            input.split(outputs.size, actualAxis)
        } else {
            input.split((split as LongArray).toIntArray(), actualAxis)
        }.map { it.asTensor() }
    }
}
