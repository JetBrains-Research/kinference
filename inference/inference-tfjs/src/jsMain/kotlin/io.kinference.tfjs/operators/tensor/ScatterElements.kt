package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.utils.getFullIndices

sealed class ScatterElements(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11, untilVersion = 16)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): ScatterElements {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ScatterElementsVer11.VERSION.asRange() -> ScatterElementsVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of ScatterElements operator: $version")
            }
        }
    }
}

class ScatterElementsVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : ScatterElements(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, 0L)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT32, TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
            IOInfo(0, ALL_DATA_TYPES, "updates", optional = false, differentiable = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11, untilVersion = 16)
        private val INFO = OperatorInfo("ScatterElements", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val indices = inputs[1]!!.data as NumberNDArrayTFJS
        val updates = inputs[2]!!.data

        require(input.type == updates.type) { "Input data type ${input.type} differs from update data type ${updates.type}." }
        require(input.rank == indices.rank && input.rank == updates.rank) {
            "Indices, updates and input must have the same rank as Input. " +
            "Indices rank=${indices.rank}. Updates rank=${updates.rank}. Input rank=${input.rank}"
        }
        require(indices.shape.contentEquals(updates.shape)) { "Indices and updates must have the same shape" }

        val actualAxis = input.indexAxis(axis)
        val axisLimit = input.shape[actualAxis]

        val output = tidyNDArray {
            val indicesForScatterNd = indices.getFullIndices(actualAxis, axisLimit, input.rank)
            input.tensorScatterUpdate(indicesForScatterNd, updates)
        }

        return listOf(output.asTensor("output"))
    }
}
