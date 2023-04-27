package io.kinference.tfjs.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.acosh
import io.kinference.operator.*
import io.kinference.tfjs.data.tensors.*

sealed class Acosh(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in AcoshVer9.VERSION.asRange() -> AcoshVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Acosh operator: $version")
            }
    }
}

class AcoshVer9(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Acosh(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("Acosh", ATTRIBUTES_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        return listOf(input.acosh().asTensor("output"))
    }
}
