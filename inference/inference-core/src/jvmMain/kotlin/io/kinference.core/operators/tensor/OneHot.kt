package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.onehot.oneHot
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto

sealed class OneHot(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): OneHot {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in OneHotVer9.VERSION.asRange() -> OneHotVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of OneHot operator: $version")
            }
        }
    }
}

class OneHotVer9 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : OneHot(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUTS_INFO = listOf(
            IOInfo(0, NUMBER_DATA_TYPES, "indices", optional = false, differentiable = false),
            IOInfo(1, NUMBER_DATA_TYPES, "depth", optional = false, differentiable = false),
            IOInfo(2, ALL_DATA_TYPES, "values", optional = true)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), required = false, default = -1L)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false, differentiable = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("OneHot", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private fun KITensor.getIntScalar(): Int {
            require(this.data.linearSize == 1) { "$name tensor must contain only one element" }

            return when (val value = this.data.singleValue()) {
                is Number -> value.toInt()
                is UInt-> value.toInt()
                is UShort -> value.toInt()
                is UByte -> value.toInt()
                is ULong -> value.toInt()
                else -> error("$name must have numeric data type, current type: ${info.type}")
            }
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val depth = inputs[1]!!.getIntScalar()
        val indices = inputs[0]!!.data as NumberNDArrayCore
        val values = inputs[2]!!.data
        return listOf(oneHot(indices, depth, values, axis).asTensor("output"))
    }
}
