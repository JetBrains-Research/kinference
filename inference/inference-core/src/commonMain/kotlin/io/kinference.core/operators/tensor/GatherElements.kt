package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIContext
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.operator.*
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class GatherElements(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when(version ?: DEFAULT_VERSION.sinceVersion) {
            in GatherElementsVer11.VERSION.asRange() -> GatherElementsVer11(attributes, inputs, outputs)
            else -> error("Unsupported version of GatherElements operator: $version")
        }
    }
}

@ExperimentalTime
class GatherElementsVer11(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : GatherElements(INFO, attributes, inputs, outputs) {
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

        private class OffsetIndexer(private val array: NDArray, indices: IntNDArray, private val axis: Int) {
            private val indicesShape = indices.shape
            private val currentDims = IntArray(array.rank)
            private val indicesPointer = indices.array.pointer()

            fun incDims() {
                currentDims[array.rank - 1] = 0
                if (array.rank == 1) return
                for (currentAxis in array.rank - 2 downTo 0) {
                    if (++currentDims[currentAxis] != indicesShape[currentAxis]) return
                    currentDims[currentAxis] = 0
                }
            }

            fun nextOffset(): Int {
                var offset = 0
                for (i in 0 until currentDims.size - 1) {
                    if (i != axis) offset += (currentDims[i] * array.strides.strides[i])
                }
                return offset + indicesPointer.getAndIncrement() * array.strides.strides[axis]
            }
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val (data, indices) = inputs.map { it!!.data }
        require(data.rank == indices.rank) { "Data and indices tensors must have the same rank" }

        val actualAxis = data.indexAxis(axis)
        val actualIndices = getIndices(indices, data.shape[actualAxis])
        val output = data.allocateNDArray(Strides(indices.shape))
        val blockSize = actualIndices.shape.last()
        val numBlocks = actualIndices.linearSize / blockSize
        val isLastDim = if (actualAxis != data.rank - 1) 1 else 0
        val indexer = OffsetIndexer(data, actualIndices, actualAxis)
        repeat(numBlocks) {
            val outputOffset = it * blockSize
            for (i in 0 until blockSize) {
                val currentOffset = i * isLastDim + indexer.nextOffset()
                output.copyFrom(outputOffset + i, data, currentOffset, currentOffset + 1)
            }
            indexer.incDims()
        }

        return listOf(output.asTensor())
    }
}
