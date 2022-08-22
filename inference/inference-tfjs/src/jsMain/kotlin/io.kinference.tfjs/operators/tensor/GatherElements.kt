package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.range
import io.kinference.tfjs.externals.extensions.*
import kotlin.time.ExperimentalTime

sealed class GatherElements(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when(version ?: DEFAULT_VERSION.sinceVersion) {
            in GatherElementsVer11.VERSION.asRange() -> GatherElementsVer11(name, attributes, inputs, outputs)
            else -> error("Unsupported version of GatherElements operator: $version")
        }
    }
}

@ExperimentalTime
class GatherElementsVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : GatherElements(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), required = false, default = 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT32, TensorProto.DataType.INT64), "indices", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("GatherElements", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val indices = inputs[1]!!.data

            // replace negative axis
            val actualAxis = input.indexAxis(axis)

            // replace negative indices
            val limitByAxis = input.shape[actualAxis]
            val indicesGreaterOrEqualZero = indices.greaterEqual(scalar(0)) // Bool tensor
            val actualIndices = indices.where(indicesGreaterOrEqualZero, indices + scalar(limitByAxis))

            // Zero pad indices to GatherND style
            val reshapedIndices = actualIndices.reshape(arrayOf(*actualIndices.shape, 1))
            val padArray = Array(reshapedIndices.rank) {
                if (it != reshapedIndices.rank - 1) {
                    arrayOf(0, 0)
                } else {
                    arrayOf(actualAxis, input.rank - actualAxis - 1)
                }
            }
            val paddedIndices = reshapedIndices.pad(padArray, 0)

            // Add relevant values to indices for GatherND
            val baseRangeShape = Array(paddedIndices.rank) { 1 }
            val baseRangePad = Array(paddedIndices.rank) { arrayOf(0,0) }

            val otherRelevantIndices = List(paddedIndices.rank - 1) { axis ->
                if (axis == actualAxis) {
                    // do nothing for operator axis
                    null
                } else {
                    // Make range for axis
                    val range = range(0, paddedIndices.shape[axis], 1, "int32")
                    val rangeShape = baseRangeShape.copyOf().apply { set(axis, paddedIndices.shape[axis]) }
                    //reshape to [1,...,paddedIndices.shape[axis],...,1]
                    // reshapedRange.rank == paddedIndices.rank
                    val reshapedRange = range.reshape(rangeShape)

                    val rangePadding = baseRangePad.copyOf().apply { set(baseRangePad.lastIndex, arrayOf(axis, input.rank - axis - 1)) }
                    // padding to [1,...,paddedIndices.shape[axis],...,input.rank]
                    val paddedRange = reshapedRange.pad(rangePadding, 0)

                    // broadcast to paddedIndices
                    paddedRange.broadcastTo(paddedIndices.shape)
                }
            }.filterNotNull().toTypedArray()

            // Adding relevant indices for GatherND
            val indicesForGatherNd = paddedIndices.add(otherRelevantIndices)

            val result = input.gatherNd(indicesForGatherNd)

            return@tidy arrayOf(result)
        }

        return listOf(outputs[0].asTensor("output"))
    }
}

