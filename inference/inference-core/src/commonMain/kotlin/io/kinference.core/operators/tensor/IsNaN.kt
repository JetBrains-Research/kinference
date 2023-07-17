package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.isNaN.isNaN
import io.kinference.operator.*
import io.kinference.primitives.types.DataType

sealed class IsNaN(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in IsNaNVer9.VERSION.asRange() -> IsNaNVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of IsNaN operator: $version")
            }
    }
}


class IsNaNVer9(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : IsNaN(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, PRIMITIVE_DATA_TYPES, "X", differentiable = true, optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, PRIMITIVE_DATA_TYPES, "Y", differentiable = true, optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("IsNaN", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val input = inputs[0]!!.data

        val output = when (input.type) {
            DataType.FLOAT -> (input as FloatNDArray).isNaN()
            DataType.DOUBLE -> (input as DoubleNDArray).isNaN()
            else -> error("Unsupported type")
        }

        return listOf(output.asTensor("Y"))
    }
}


