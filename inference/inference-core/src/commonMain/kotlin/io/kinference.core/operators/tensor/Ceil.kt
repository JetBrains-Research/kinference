package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.ceil.ceil
import io.kinference.operator.*
import io.kinference.primitives.types.DataType

sealed class Ceil(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in CeilVer6.VERSION.asRange() -> CeilVer6(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Ceil operator: $version")
            }
    }
}


class CeilVer6(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Ceil(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, PRIMITIVE_DATA_TYPES, "X", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, PRIMITIVE_DATA_TYPES, "Y", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Ceil", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data as NumberNDArrayCore
        val output = when(input.type) {
            DataType.FLOAT -> (input as FloatNDArray).ceil()
            DataType.DOUBLE -> (input as DoubleNDArray).ceil()
            else -> error("Unsupported input type in Ceil operator, current type ${input.type}")
        }
        return listOf(output.asTensor("Y"))
    }
}


