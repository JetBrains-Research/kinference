package io.kinference.core.operators.tensor

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.computeBlockSize
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class ScatterND(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
            IOInfo(0, ALL_DATA_TYPES, "updates", optional = false, differentiable = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        private val VERSION = VersionInfo(sinceVersion = 11, untilVersion = 16)
        private val INFO = OperatorInfo("ScatterND", emptyList(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in VERSION.asRange() -> ScatterND(attributes, inputs, outputs)
            else -> error("Unsupported version of ScatterND operator: $version")
        }

        private fun getActualIndices(input: NDArray, indices: LongNDArray, kDim: Int): IntArray {
            val inputStrides = input.strides.strides
            val numBlocks = indices.linearSize / kDim
            val indicesPointer = indices.array.pointer()
            return IntArray(numBlocks) {
                var acc = 0
                for (i in 0 until kDim) acc += indicesPointer.getAndIncrement().toInt() * inputStrides[i]
                acc
            }
        }
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!.data.toMutable()
        val indices = inputs[1]!!.data as LongNDArray
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
