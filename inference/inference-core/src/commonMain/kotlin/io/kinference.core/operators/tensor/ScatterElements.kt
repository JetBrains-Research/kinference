package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.stack
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.InlineInt

sealed class ScatterElements(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
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

        private suspend fun getIndices(indices: NDArray, axisLimit: Int): IntNDArray {
            if (indices !is IntNDArray && indices !is LongNDArray) error("Indices type must be either Long or Int. Current type = ${indices.type}")

            fun checkIndex(index: Int, axisLimit: Int): Int = if (index >= 0) index else index + axisLimit

            return if (indices is IntNDArray) {
                indices.map (object : IntMap {
                    override fun apply(value: Int): Int = checkIndex(value, axisLimit)
                })
            } else {
                indices as LongNDArray
                val pointer = indices.array.pointer()
                val typedLambda: (InlineInt) -> Int = { checkIndex(pointer.getAndIncrement().toInt(), axisLimit) }
                IntNDArray(indices.shape, typedLambda)
            }
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    private class DimensionStepCounter(val targetArrayShape: IntArray, val numDims: Int) {
        private val dimStepCounter = IntArray(numDims)

        operator fun get(i: Int): Int = dimStepCounter[i]

        fun update() {
            for (j in numDims - 1 downTo 0) {
                dimStepCounter[j] += 1
                require(dimStepCounter[j] <= targetArrayShape[j]) { "Cannot update more elements than $j-th dimension of the input array has. Max = ${targetArrayShape[j]}" }

                if (dimStepCounter[j] < targetArrayShape[j]) break
                dimStepCounter[j] = 0
            }
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data.toMutable()
        val indicesInput = inputs[1]!!.data

        val updates = inputs[2]!!.data
        val actualAxis = input.indexAxis(axis)

        require(input.type == updates.type) { "Input data type ${input.type} differs from update data type ${updates.type}." }
        require(input.rank == indicesInput.rank && input.rank == updates.rank) {
            "Indices, updates and input must have the same rank as Input. " +
            "Indices rank=${indicesInput.rank}. Updates rank=${updates.rank}. Input rank=${input.rank}"
        }
        require(indicesInput.shape.contentEquals(updates.shape)) { "Indices and updates must have the same shape" }

        val indices = getIndices(indicesInput, input.shape[actualAxis])
        val inputStrides = input.strides.strides

        val counter = DimensionStepCounter(updates.shape, input.rank)
        val indicesPointer = indices.array.pointer()
        for (i in 0 until indices.linearSize) {
            val targetIndex = indicesPointer.getAndIncrement()
            val dstOffset = inputStrides.foldIndexed(0) { index, acc, stride ->
                acc + stride * (if (index == actualAxis) targetIndex else counter[index])
            }
            input.copyFrom(dstOffset, updates, i, i + 1)
            counter.update()
        }
        return listOf(input.asTensor("output"))
    }
}
