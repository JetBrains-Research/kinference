package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.floor.floor
import io.kinference.operator.*
import io.kinference.primitives.types.DataType

sealed class Floor(
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
                in FloorVer6.VERSION.asRange() -> FloorVer6(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Floor operator: $version")
            }
    }
}


class FloorVer6(
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

        //Realized the latest version, but there is backward compatibility between operators
        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Floor", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data as NumberNDArrayCore
        val output = when(input.type) {
            DataType.FLOAT -> (input as FloatNDArray).floor()
            DataType.DOUBLE -> (input as DoubleNDArray).floor()
            else -> error("Unsupported input type in Floor operator, current type ${input.type}")
        }
        return listOf(output.asTensor("Y"))
    }
}


