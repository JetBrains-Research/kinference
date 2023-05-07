package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Flatten(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Flatten {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in FlattenVer1.VERSION.asRange() -> FlattenVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Constant operator: $version")
            }
        }
    }
}

class FlattenVer1(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Flatten(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, differentiable = true),
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = true))

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), default = 1, required = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Flatten", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    val axis: Int by attribute { it: Number -> it.toInt() }

    private fun NDArrayTFJS.flattenedShape(axis: Int): IntArray {
        val firstDimension = computeBlockSize(toDim = axis)
        val secondDimension = computeBlockSize(fromDim = axis)

        return intArrayOf(firstDimension, secondDimension)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val actualAxis = input.indexAxis(axis)

        val newShape = input.flattenedShape(actualAxis)
        val result = tidyNDArray { input.reshape(newShape) }

        return listOf(result.asTensor("output"))
    }
}
