package io.kinference.operators.tensor

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.computeBlockSize
import io.kinference.operators.*
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class ScatterND(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
            IOInfo(0, ALL_DATA_TYPES, "updates", optional = false, differentiable = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        private val INFO = OperatorInfo("ScatterND", emptyList(), INPUTS_INFO, OUTPUTS_INFO)

        private fun LongNDArray.toIntNDArray(): IntNDArray {
            val indicesPointer = this.array.pointer()
            return IntNDArray(this.shape) { indicesPointer.getAndIncrement().toInt() }
        }

        private fun getActualIndices(input: NDArray, indices: IntNDArray, kDim: Int): IntArray {
            val inputStrides = input.strides.strides
            val numBlocks = indices.linearSize / kDim
            val indicesPointer = indices.array.pointer()
            return IntArray(numBlocks) {
                var acc = 0
                for (i in 0 until kDim) acc += indicesPointer.getAndIncrement() * inputStrides[i]
                acc
            }
        }
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val input = inputs[0]!!.data.toMutable()
        val indices = (inputs[1]!!.data as LongNDArray).toIntNDArray()
        val updates = inputs[2]!!.data

        val kDim = indices.shape.last()
        val blockSize = input.computeBlockSize(fromDim = kDim)
        val srcUpdateOffsets = getActualIndices(input, indices, kDim)

        for ((i, offset) in srcUpdateOffsets.withIndex()) {
            val srcOff = i * blockSize
            input.copyFrom(offset, updates, srcOff, srcOff + blockSize)
        }

        return listOf(input.asTensor("output"))
    }
}
