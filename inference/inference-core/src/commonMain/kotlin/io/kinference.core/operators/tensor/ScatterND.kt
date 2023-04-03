package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto

sealed class ScatterND(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11, untilVersion = 16)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ScatterNDVer11.VERSION.asRange() -> ScatterNDVer11(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

class ScatterNDVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : ScatterND(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
            IOInfo(0, ALL_DATA_TYPES, "updates", optional = false, differentiable = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11, untilVersion = 16)
        private val INFO = OperatorInfo("ScatterND", emptyList(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

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

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
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
