package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.utils.inlines.InlineInt

sealed class Constant(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ConstantVer1.VERSION.asRange() -> ConstantVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}


class ConstantVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Constant(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("value", setOf(AttributeProto.AttributeType.TENSOR), false),
            AttributeInfo("value_float", setOf(AttributeProto.AttributeType.FLOAT), false),
            AttributeInfo("value_floats", setOf(AttributeProto.AttributeType.FLOATS), false),
            AttributeInfo("value_int", setOf(AttributeProto.AttributeType.INT), false),
            AttributeInfo("value_ints", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("value_string", setOf(AttributeProto.AttributeType.STRING), false),
            AttributeInfo("value_strings", setOf(AttributeProto.AttributeType.STRINGS), false)
            //TODO: sparse tensor values
        )

        private val INPUTS_INFO = emptyList<IOInfo>()

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Constant", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        //only one of all attributes is not null
        val (name, value) = ATTRIBUTES_INFO.map { it.name to getAttributeOrNull<Any?>(it.name) }.single { it.second != null }

        @Suppress("UNCHECKED_CAST")
        val result = when (name) {
            "value" -> value
            "value_float" -> FloatNDArray.scalar(value as Float).asTensor()
            "value_floats" -> {
                value as FloatArray
                FloatNDArray(intArrayOf(value.size)) { it: InlineInt -> value[it.value] }.asTensor()
            }
            "value_int" -> LongNDArray.scalar(value as Long).asTensor()
            "value_ints" -> {
                value as LongArray
                LongNDArray(intArrayOf(value.size)) { it: InlineInt -> value[it.value] }.asTensor()
            }
            "value_string" -> StringNDArray.scalar(value!! as String).asTensor()
            "value_strings" -> {
                value as List<String>
                StringNDArray(intArrayOf(value.size)) { value[it] }.asTensor()
            }
            else -> error("Unsupported data type")
        } as KITensor
        return listOf(result)
    }
}
