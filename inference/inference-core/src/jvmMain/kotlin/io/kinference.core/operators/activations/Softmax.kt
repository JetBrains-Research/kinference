package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto

sealed class Softmax(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 13, untilVersion = 23)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in SoftmaxVer1.VERSION.asRange() -> SoftmaxVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Softmax operator: $version")
        }
    }
}

//only for float and double types

class SoftmaxVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Softmax(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = -1)
        )

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 13, untilVersion = 23)
        private val INFO = OperatorInfo("Softmax", ATTRIBUTES_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override suspend fun activate(input: NDArrayCore, contexts: Contexts<KIONNXData<*>>): NDArrayCore {
        input as NumberNDArrayCore
        return input.softmax(axis)
    }
}
