package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.rem.pythonRem
import io.kinference.ndarray.extensions.rem.rem
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto

sealed class Mod(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ModVer10.VERSION.asRange() -> ModVer10(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Mod operator: $version")
            }
    }
}


class ModVer10(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Mod(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = NUMBER_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("fmod", setOf(AttributeProto.AttributeType.INT), required = false, default = 0),
        )

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("Mod", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }

    private val fmod: Boolean by attribute { it: Number -> it.toInt() != 0 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val left = inputs[0]!!.data as NumberNDArrayCore
        val right = inputs[1]!!.data as NumberNDArrayCore

        require(left.type == right.type)
        val inputType = left.type

        val output = if (fmod) {
            left.rem(right)
        } else {
            when (inputType) {
                DataType.BYTE -> (left as ByteNDArray).pythonRem(right as ByteNDArray)
                DataType.SHORT -> (left as ShortNDArray).pythonRem(right as ShortNDArray)
                DataType.INT -> (left as IntNDArray).pythonRem(right as IntNDArray)
                DataType.LONG -> (left as LongNDArray).pythonRem(right as LongNDArray)
                DataType.UBYTE -> (left as UByteNDArray).rem(right as UByteNDArray)
                DataType.USHORT -> (left as UShortNDArray).rem(right as UShortNDArray)
                DataType.UINT -> (left as UIntNDArray).rem(right as UIntNDArray)
                DataType.ULONG -> (left as ULongNDArray).rem(right as ULongNDArray)
                else -> error("Operator Mod with attribute fmod=0 supports only Int tensors, current type is $inputType")
            }
        }

        return listOf(output.asTensor("C"))
    }
}
