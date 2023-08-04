package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.reciprocal
import io.kinference.operator.*

sealed class Reciprocal(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Reciprocal {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ReciprocalVer6.VERSION.asRange() -> ReciprocalVer6(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Reciprocal operator: $version")
            }
        }
    }
}

class ReciprocalVer6 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Reciprocal(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, FLOAT_DATA_TYPES, "X", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, FLOAT_DATA_TYPES, "Y", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Reciprocal", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val output = when (val input = inputs[0]!!.data as NumberNDArrayCore) {
            is FloatNDArray -> input.reciprocal()
            is DoubleNDArray -> input.reciprocal()
            else -> error("Unsupported data type: ${input.type}")
        }
        return listOf(output.asTensor("Y"))
    }
}
