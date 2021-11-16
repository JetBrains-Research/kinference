package io.kinference.core.operators.tensor

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.core.operators.*
import io.kinference.core.operators.VersionInfo.Companion.asRange
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto

sealed class Constant(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        operator fun invoke(version: Int, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version) {
            in ConstantVer1.VERSION.asRange() -> ConstantVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

@ExperimentalTime
class ConstantVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Constant(INFO, attributes, inputs, outputs) {
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


    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        //only one of all attributes is not null
        val (name, value) = ATTRIBUTES_INFO.map { it.name to getAttributeOrNull<Any?>(it.name) }.single { it.second != null }

        @Suppress("UNCHECKED_CAST")
        val result = when (name) {
            "value" -> value
            "value_float" -> FloatNDArray.scalar(value as Float).asTensor()
            "value_floats" -> {
                value as FloatArray
                FloatNDArray(intArrayOf(value.size)) { value[it] }.asTensor()
            }
            "value_int" -> LongNDArray.scalar(value as Long).asTensor()
            "value_ints" -> {
                value as LongArray
                LongNDArray(intArrayOf(value.size)) { value[it] }.asTensor()
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
