package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Mod(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ModVer10.VERSION.asRange() -> ModVer10(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Mod operator: $version")
            }
    }
}

class ModVer10(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Mod(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = NUMBER_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("fmod", setOf(AttributeProto.AttributeType.INT), required = false, default = 0),
        )

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("Mod", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }

    private val fmod: Boolean by attribute { it: Number -> it.toInt() != 0 }
    private val modFunction = if (fmod) NumberNDArrayTFJS::fmod else NumberNDArrayTFJS::mod

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val left = inputs[0]!!.data as NumberNDArrayTFJS
        val right = inputs[1]!!.data as NumberNDArrayTFJS

        require(left.type == right.type) { "Input types must have same data type, current ${left.type} != ${right.type}" }
        val inputType = left.type

        require(fmod || inputType != DataType.FLOAT) { "Operator Mod with attribute fmod=0 supports only Int tensors, current type is $inputType" }

        val output = this.modFunction(left, right)

        return listOf(output.asTensor("C"))
    }
}
