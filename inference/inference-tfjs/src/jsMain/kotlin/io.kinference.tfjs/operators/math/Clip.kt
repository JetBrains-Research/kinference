package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.clip
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Clip(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
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

        private fun getMinMax(type: DataType): Pair<Number, Number> = when(type) {
            DataType.FLOAT -> Float.NEGATIVE_INFINITY to Float.POSITIVE_INFINITY
            DataType.DOUBLE -> Double.NEGATIVE_INFINITY to Double.POSITIVE_INFINITY
            DataType.INT, DataType.UINT -> Int.MIN_VALUE to Int.MAX_VALUE
            DataType.BYTE, DataType.UBYTE -> Byte.MIN_VALUE to Byte.MAX_VALUE
            DataType.SHORT, DataType.USHORT -> Short.MIN_VALUE to Short.MAX_VALUE
            DataType.LONG, DataType.ULONG -> Long.MIN_VALUE to Long.MAX_VALUE
            else -> error("Unsupported input type in Clip operator, current type $type")
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val min = inputs.getOrNull(1)?.data as? NumberNDArrayTFJS
        val max = inputs.getOrNull(2)?.data as? NumberNDArrayTFJS

        val (defaultMin, defaultMax) = getMinMax(input.type)
        val minScalar = min?.singleValue() ?: defaultMin
        val maxScalar = max?.singleValue() ?: defaultMax

        return listOf(input.clip(minScalar, maxScalar).asTensor("output"))
    }
}
