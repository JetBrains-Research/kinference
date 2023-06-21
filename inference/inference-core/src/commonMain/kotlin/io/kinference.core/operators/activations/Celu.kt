package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.activations.celu.celu
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto

sealed class Celu(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Activation(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 12)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in CeluVer12.VERSION.asRange() -> CeluVer12(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Celu operator: $version")
            }
    }
}


class CeluVer12(
    name: String,
    attributes: Map<String, Attribute<Any>> = emptyMap(),
    inputs: List<String>,
    outputs: List<String>
) : Celu(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTE_INFO = listOf(AttributeInfo("alpha", setOf(AttributeProto.AttributeType.FLOAT), default = 1f))

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("Celu", ATTRIBUTE_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val alpha: Float by attribute { it: Number -> it.toFloat() }

    override suspend fun activate(input: NDArrayCore, contexts: Contexts<KIONNXData<*>>): NDArrayCore {
        return (input as FloatNDArray).celu(alpha)
    }
}

