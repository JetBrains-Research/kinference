package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.createArray
import io.kinference.ndarray.extensions.isScalar
import io.kinference.operators.*
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class NonZero(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "X", optional = false, differentiable = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "Y", optional = false, differentiable = false))

        private val INFO = OperatorInfo("NonZero", emptyMap(), INPUTS_INFO, OUTPUTS_INFO)

        private fun NDArray.indicesIterator() = NDIndexIterator(this)
    }

    private class NDIndexIterator(array: NDArray) : Iterator<LongArray> {
        private val shape = array.shape
        private val indexSize = shape.size
        private val maxElements = array.linearSize
        private var elementsCounter = 0
        private var currentIndex = LongArray(indexSize).apply { this[lastIndex] = -1L }

        override fun hasNext(): Boolean = elementsCounter < maxElements

        override fun next(): LongArray {
            for (idx in indexSize - 1 downTo 0) {
                if (currentIndex[idx] != (shape[idx] - 1).toLong()) {
                    currentIndex[idx]++
                    break
                }
                currentIndex[idx] = 0
            }
            return currentIndex
        }
    }


    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val input = inputs[0]!!.data
        return listOf(input.nonZero().asTensor("Y"))
    }
}
