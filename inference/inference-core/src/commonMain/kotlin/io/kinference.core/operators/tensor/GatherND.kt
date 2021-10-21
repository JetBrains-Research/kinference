package io.kinference.core.operators.tensor

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.NDArray
import io.kinference.ndarray.extensions.computeBlockSize
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class GatherND(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = listOf(AttributeInfo("batch_dims", setOf(AttributeProto.AttributeType.INT), false, 0L))

        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        private val INFO = OperatorInfo("GatherND", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)

        private fun NDArray.getOffsetsFromIndices(indices: LongNDArray, batchDims: Int): IntArray {
            val indexSize = indices.shape.last()
            val numBlocks = indices.computeBlockSize(toDim = indices.rank - 1)
            val numBatches = this.computeBlockSize(toDim = batchDims)
            val numBlocksPerBatch = numBlocks / numBatches
            val batchSize = this.computeBlockSize(fromDim = batchDims)
            val blockDimsSizes = IntArray(indexSize) { this.computeBlockSize(fromDim = batchDims + it + 1) }
            val indicesPointer = indices.array.pointer()
            return IntArray(numBlocks) { block ->
                val batchIdx = block / numBlocksPerBatch
                var blockOffset = 0
                for (idx in 0 until indexSize) {
                    val currentIdx = indicesPointer.getAndIncrement().toInt()
                    val maxIdx = shape[batchDims + idx]
                    blockOffset += blockDimsSizes[idx] * (if (currentIdx < 0) currentIdx + maxIdx else currentIdx)
                }
                batchIdx * batchSize + blockOffset
            }
        }

        private fun inferOutputShape(inputShape: IntArray, indicesShape: IntArray, batchDims: Int): IntArray {
            val lastIndicesDim = indicesShape.last() + batchDims
            return (indicesShape.dropLast(1) + inputShape.drop(lastIndicesDim)).toIntArray()
        }
    }

    private val batchDims: Int by attribute("batch_dims") { it: Number -> it.toInt() }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!.data
        val indices = inputs[1]!!.data as LongNDArray
        val blockSize = input.computeBlockSize(fromDim = batchDims + indices.shape.last())
        val offsets = input.getOffsetsFromIndices(indices, batchDims)
        val outputShape = inferOutputShape(input.shape, indices.shape, batchDims)
        val output = input.allocateNDArray(Strides(outputShape))
        for ((i, offset) in offsets.withIndex()) {
            output.copyFrom(i * blockSize, input, offset, offset + blockSize)
        }
        return listOf(output.asTensor("output"))
    }
}
