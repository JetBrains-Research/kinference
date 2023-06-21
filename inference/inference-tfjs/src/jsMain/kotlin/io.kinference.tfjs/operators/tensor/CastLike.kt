package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class CastLike(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 15, untilVersion = 19)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in CastLikeVer15.VERSION.asRange() -> CastLikeVer15(name, attributes, inputs, outputs)
                else -> error("Unsupported version of CastLike operator: $version")
            }
    }
}

class CastLikeVer15(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : CastLike(name, INFO, attributes, inputs, outputs) {

    companion object {
        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, PRIMITIVE_DATA_TYPES, "input", optional = false),
            IOInfo(1, PRIMITIVE_DATA_TYPES, "target_type", optional = false),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, PRIMITIVE_DATA_TYPES, "output", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 15, untilVersion = 19)
        private val INFO = OperatorInfo("CastLike", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val toType = inputs[1]!!.info.type

        val output = CastVer6.castTo(input, toType)

        return listOf(output.asTensor("Y"))
    }
}
