package io.kinference.core.operators.bitwise

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.bitwise.or.bitOr
import io.kinference.operator.*
import io.kinference.primitives.types.DataType

sealed class BitwiseOr(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 18)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in BitwiseOrVer18.VERSION.asRange() -> BitwiseOrVer18(name, attributes, inputs, outputs)
                else -> error("Unsupported version of BitwiseOr operator: $version")
            }
    }
}


class BitwiseOrVer18(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : BitwiseOr(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = UINT_DATA_TYPES + INT_DATA_TYPES

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false, differentiable = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false, differentiable = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 18)
        private val INFO = OperatorInfo("BitwiseOr", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val left = inputs[0]!!.data
        val right = inputs[1]!!.data

        require(left.type == right.type) { "" }

        val dest = when (left.type) {
            DataType.BYTE -> (left as ByteNDArray).bitOr(right as ByteNDArray)
            DataType.SHORT -> (left as ShortNDArray).bitOr(right as ShortNDArray)
            DataType.INT -> (left as IntNDArray).bitOr(right as IntNDArray)
            DataType.LONG -> (left as LongNDArray).bitOr(right as LongNDArray)
            DataType.UBYTE -> (left as UByteNDArray).bitOr(right as UByteNDArray)
            DataType.USHORT -> (left as UShortNDArray).bitOr(right as UShortNDArray)
            DataType.UINT -> (left as UIntNDArray).bitOr(right as UIntNDArray)
            DataType.ULONG -> (left as ULongNDArray).bitOr(right as ULongNDArray)
            else -> error("Unsupported input type, current type ${left.type}")
        }

        return listOf(dest.asTensor("C"))
    }
}
