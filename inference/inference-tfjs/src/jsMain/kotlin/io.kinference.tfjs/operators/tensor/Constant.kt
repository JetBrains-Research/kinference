package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.toIntArray
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Constant(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ConstantVer1.VERSION.asRange() -> ConstantVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Constant operator: $version")
            }
    }
}

class ConstantVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Constant(name, INFO, attributes, inputs, outputs) {
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


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        //only one of all attributes is not null
        val (name, value) = ATTRIBUTES_INFO.map { it.name to getAttributeOrNull<Any?>(it.name) }.single { it.second != null }

        val result = when (name) {
            "value" -> (value as TFJSTensor).data.clone().asTensor()
            "value_float" -> NDArrayTFJS.floatScalar(value as Float).asTensor()
            "value_floats" -> {
                value as FloatArray
                NDArrayTFJS.float(value, arrayOf(value.size)).asTensor()
            }

            "value_int" -> NDArrayTFJS.intScalar(value as Int).asTensor()
            "value_ints" -> {
                value as LongArray
                NDArrayTFJS.int(value.toIntArray(), arrayOf(value.size)).asTensor()
            }

            "value_string", "value_strings" -> error("String not supported in Constant now")
            else -> error("Unsupported data type")
        }
        return listOf(result)
    }
}
