package io.kinference.webgpu.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.ndarray.reversed
import io.kinference.ndarray.toIntArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.webgpu.ndarray.NDArrayInfo
import io.kinference.webgpu.operators.common.UnaryOperator
import io.kinference.webgpu.operators.common.shapeToWorkSize
import io.kinference.webgpu.utils.divUp

sealed class Transpose(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : UnaryOperator(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in TransposeVer1.VERSION.asRange() -> TransposeVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

class TransposeVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Transpose(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("perm", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "transposed", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Transpose", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val perm: IntArray? by attributeOrNull { it: LongArray? -> it?.toIntArray() }

    private fun actualPerm(input: NDArrayInfo) = perm ?: input.shape.indices.reversed()

    private fun blockTranspose(input: NDArrayInfo) = actualPerm(input).last() == input.shape.indices.last

    override fun outputInfo(inputInfo: List<NDArrayInfo?>): List<NDArrayInfo?> {
        val input = inputInfo[0]!!
        val actualPerm: IntArray = perm ?: input.shape.indices.reversed()
        val outputShape = IntArray(input.rank) { index -> input.shape[actualPerm[index]] }
        return listOf(NDArrayInfo(outputShape, input.type))
    }

    override fun workGroupSize(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>): IntArray =
        if (blockTranspose(inputInfo[0]!!)) intArrayOf(128, 1, 1) else intArrayOf(16, 16, 1)

    override fun dispatchSize(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>, workGroupSize: IntArray): IntArray {
        val input = inputInfo[0]!!
        val globalWorkSize = if (blockTranspose(input)) {
            shapeToWorkSize(input.shape)
        } else {
            val xIndex = input.shape.indices.last
            val yIndex = actualPerm(input).last()
            intArrayOf(
                input.shape[xIndex],
                input.shape[yIndex],
                input.shape.filterIndexed { index, _ -> index != xIndex && index != yIndex }.fold(1, Int::times))
        }
        return (globalWorkSize zip workGroupSize).map { (dim, workSize) -> dim divUp workSize }.toIntArray()
    }

    override fun createShader(inputInfo: List<NDArrayInfo?>, outputInfo: List<NDArrayInfo?>): String {
        TODO("Not yet implemented")
    }
}
