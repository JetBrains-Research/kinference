package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.clip.clip
import io.kinference.operator.*
import io.kinference.primitives.types.DataType

sealed class Clip(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ClipVer11.VERSION.asRange() -> ClipVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Clip operator: $version")
            }
    }
}

class ClipVer11 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Clip(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, NUMBER_DATA_TYPES, "input", optional = false),
            IOInfo(1, NUMBER_DATA_TYPES, "min", optional = true),
            IOInfo(2, NUMBER_DATA_TYPES, "max", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, NUMBER_DATA_TYPES, "output", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("Clip", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data as NumberNDArrayCore
        val min = inputs.getOrNull(1)?.data as? NumberNDArrayCore
        val max = inputs.getOrNull(2)?.data as? NumberNDArrayCore

        val minScalar = min?.singleValue()
        val maxScalar = max?.singleValue()

        val output = when(input.type) {
            DataType.FLOAT -> (input as FloatNDArray).clip(minScalar as? Float, maxScalar as? Float)
            DataType.DOUBLE -> (input as DoubleNDArray).clip(minScalar as? Double, maxScalar as? Double)
            DataType.INT -> (input as IntNDArray).clip(minScalar as? Int, maxScalar as? Int)
            DataType.UINT -> (input as UIntNDArray).clip(minScalar as? UInt, maxScalar as? UInt)
            DataType.BYTE -> (input as ByteNDArray).clip(minScalar as? Byte, maxScalar as? Byte)
            DataType.UBYTE -> (input as UByteNDArray).clip(minScalar as? UByte, maxScalar as? UByte)
            DataType.SHORT -> (input as ShortNDArray).clip(minScalar as? Short, maxScalar as? Short)
            DataType.USHORT -> (input as UShortNDArray).clip(minScalar as? UShort, maxScalar as? UShort)
            DataType.LONG -> (input as LongNDArray).clip(minScalar as? Long, maxScalar as? Long)
            DataType.ULONG -> (input as ULongNDArray).clip(minScalar as? ULong, maxScalar as? ULong)
            else -> error("Unsupported input type in Clip operator, current type ${input.type}")
        }
        return listOf(output.asTensor("output"))
    }
}
