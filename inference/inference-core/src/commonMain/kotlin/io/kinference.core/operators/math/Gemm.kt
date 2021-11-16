package io.kinference.core.operators.math

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.unsqueezeFirst
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.gemm
import io.kinference.core.operators.*
import io.kinference.primitives.types.DataType
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class Gemm(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in GemmVer11.VERSION.asRange() -> GemmVer11(attributes, inputs, outputs)
            else -> error("Unsupported version of Gemm operator: $version")
        }
    }
}

@ExperimentalTime
class GemmVer11(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Gemm(INFO, attributes, inputs, outputs) {
    private val alpha: Double by attribute { it: Number -> it.toDouble() }
    private val beta: Double by attribute { it: Number -> it.toDouble() }

    private val transA: Boolean by attribute { it: Number -> it.toInt() != 0 }
    private val transB: Boolean by attribute { it: Number -> it.toInt() != 0 }

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.BFLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("alpha", setOf(AttributeProto.AttributeType.FLOAT), false, 1.0),
            AttributeInfo("beta", setOf(AttributeProto.AttributeType.FLOAT), false, 1.0),
            AttributeInfo("transA", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("transB", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, TYPE_CONSTRAINTS, "C", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("Gemm", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private fun getDest(array: NDArray?, type: DataType, targetShape: IntArray): MutableNDArray {
            if (array == null) return allocateNDArray(type, Strides(targetShape))
            if (array.shape.contentEquals(targetShape)) return array.toMutable()

            val dstArray = allocateNDArray(type, Strides(targetShape)) as MutableNumberNDArray
            val unsqueezedShape = unsqueezeFirst(array.shape, targetShape.size)

            if (targetShape[1] != unsqueezedShape[1] && unsqueezedShape[1] == 1) {
                val targetBlockSize = targetShape[1]
                for (i in 0 until unsqueezedShape[0]) {
                    val dstOffsetBase = i * targetBlockSize
                    dstArray.fillByArrayValue(array, i, dstOffsetBase, dstOffsetBase + targetBlockSize)
                }
            } else {
                dstArray.copyFrom(0, array)
            }

            for (i in 1 until targetShape[0]) dstArray.copyFrom(i * targetShape[1], dstArray, 0, targetShape[1])
            return dstArray
        }
    }

    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val a = inputs[0]!!.data as NumberNDArray
        val b = inputs[1]!!.data as NumberNDArray

        val m = if (!transA) a.shape[0] else a.shape[1]
        val n = if (!transB) b.shape[1] else b.shape[0]
        val k = if (!transA) a.shape[1] else a.shape[0]

        val c = getDest(inputs.getOrNull(2)?.data, a.type, intArrayOf(m, n))
        gemm(m, n, k, alpha, a, b, beta, c, transposeA = transA, transposeB = transB)

        return listOf(c.asTensor())
    }
}
