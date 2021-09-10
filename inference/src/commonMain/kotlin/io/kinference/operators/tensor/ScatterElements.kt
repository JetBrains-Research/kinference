package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class ScatterElements(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
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

        private val INFO = OperatorInfo("ScatterElements", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        private fun getIndices(indices: NDArray, axisLimit: Int): IntNDArray {
            if (indices !is IntNDArray && indices !is LongNDArray) error("Indices type must be either Long or Int. Current type = ${indices.type}")

            fun checkIndex(index: Int, axisLimit: Int): Int = if (index >= 0) index else index + axisLimit

            return if (indices is IntNDArray) {
                indices.map (object : IntMap {
                    override fun apply(value: Int): Int = checkIndex(value, axisLimit)
                }) as IntNDArray
            } else {
                indices as LongNDArray
                val pointer = indices.array.pointer()
                IntNDArray(indices.shape) { checkIndex(pointer.getAndIncrement().toInt(), axisLimit) }
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

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
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
