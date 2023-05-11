package io.kinference.core.operators.bitwise

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.bitwise.not.bitNot
import io.kinference.operator.*
import io.kinference.primitives.types.DataType

sealed class BitwiseNot(
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
                in BitwiseNotVer18.VERSION.asRange() -> BitwiseNotVer18(name, attributes, inputs, outputs)
                else -> error("Unsupported version of BitNot operator: $version")
            }
    }
}


class BitwiseNotVer18(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : BitwiseNot(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = UINT_DATA_TYPES + INT_DATA_TYPES

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false, differentiable = false),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false, differentiable = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 18)
        private val INFO = OperatorInfo("BitNot", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data

        val dest = when (input.type) {
            DataType.BYTE -> (input as ByteNDArray).bitNot()
            DataType.SHORT -> (input as ShortNDArray).bitNot()
            DataType.INT -> (input as IntNDArray).bitNot()
            DataType.LONG -> (input as LongNDArray).bitNot()
            DataType.UBYTE -> (input as UByteNDArray).bitNot()
            DataType.USHORT -> (input as UShortNDArray).bitNot()
            DataType.UINT -> (input as UIntNDArray).bitNot()
            DataType.ULONG -> (input as ULongNDArray).bitNot()
            else -> error("Unsupported input type, current type ${input.type}")
        }

        return listOf(dest.asTensor("C"))
    }
}
