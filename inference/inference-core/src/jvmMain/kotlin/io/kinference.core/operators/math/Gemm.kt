package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.unsqueezeFirst
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.gemm
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class Gemm(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    private val alpha: Double by attribute { it: Number -> it.toDouble() }
    private val beta: Double by attribute { it: Number -> it.toDouble() }

    private val transA: Boolean by attribute { it: Number -> it.toInt() != 0 }
    private val transB: Boolean by attribute { it: Number -> it.toInt() != 0 }

    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in GemmVer9.VERSION.asRange() -> GemmVer9(name, attributes, inputs, outputs)
            in GemmVer11.VERSION.asRange() -> GemmVer11(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Gemm operator: $version")
        }
    }

    protected suspend fun getDest(array: NDArrayCore, type: DataType, targetShape: IntArray): MutableNDArrayCore {
        if (array.shape.contentEquals(targetShape)) return array.toMutable()

        val dstArray = allocateNDArray(type, Strides(targetShape)) as MutableNumberNDArrayCore
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

    protected suspend fun <D : ONNXData<*, *>> apply(inputs: List<KITensor?>, optionalBias: Boolean): List<KITensor?> {
        val a = inputs[0]!!.data as NumberNDArrayCore
        val b = inputs[1]!!.data as NumberNDArrayCore

        val m = if (!transA) a.shape[0] else a.shape[1]
        val n = if (!transB) b.shape[1] else b.shape[0]
        val k = if (!transA) a.shape[1] else a.shape[0]

        val targetShape = intArrayOf(m, n)
        val bias = if (optionalBias) {
            inputs.getOrNull(2)?.data ?: allocateNDArray(a.type, targetShape)
        } else {
            inputs[2]!!.data
        } as NumberNDArrayCore

        val c = getDest(bias, a.type, intArrayOf(m, n))
        gemm(m, n, k, alpha, a, b, beta, c, transposeA = transA, transposeB = transB)

        return listOf(c.asTensor())
    }
}

class GemmVer9(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Gemm(name, INFO, attributes, inputs, outputs) {
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
            IOInfo(2, TYPE_CONSTRAINTS, "C", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9, untilVersion = 11)
        private val INFO = OperatorInfo("Gemm", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        return apply<ONNXData<*, *>>(inputs, INPUTS_INFO[2].optional)
    }
}

class GemmVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Gemm(name, INFO, attributes, inputs, outputs) {
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
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        return apply<ONNXData<*, *>>(inputs, INPUTS_INFO[2].optional)
    }
}
