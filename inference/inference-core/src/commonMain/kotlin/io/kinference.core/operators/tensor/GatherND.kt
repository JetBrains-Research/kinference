package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class GatherND(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): GatherND {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in GatherNDVer11.VERSION.asRange() -> GatherNDVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of GatherND operator: $version")
            }
        }
    }
}

class GatherNDVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : GatherND(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = listOf(AttributeInfo("batch_dims", setOf(AttributeProto.AttributeType.INT), false, 0L))

        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("GatherND", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

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

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data
        val indices = inputs[1]!!.data as LongNDArray
        val blockSize = input.computeBlockSize(fromDim = batchDims + indices.shape.last())
        val offsets = input.getOffsetsFromIndices(indices, batchDims)
        val outputShape = inferOutputShape(input.shape, indices.shape, batchDims)
        val output = allocateNDArray(input.type, Strides(outputShape))
        for ((i, offset) in offsets.withIndex()) {
            output.copyFrom(i * blockSize, input, offset, offset + blockSize)
        }
        return listOf(output.asTensor("output"))
    }
}
