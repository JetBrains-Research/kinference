package io.kinference.webgpu.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.toIntArray
import io.kinference.operator.*
import io.kinference.operator.Operator.Companion.ALL_DATA_TYPES
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.AttributeProto
import io.kinference.webgpu.graph.WebGPUContext
import io.kinference.webgpu.ndarray.*
import io.kinference.webgpu.tensor.WebGPUTensor

sealed class Constant(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<WebGPUTensor, WebGPUTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ConstantVer1.VERSION.asRange() -> ConstantVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

class ConstantVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Constant(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("value", setOf(AttributeProto.AttributeType.TENSOR), false),
            AttributeInfo("value_float", setOf(AttributeProto.AttributeType.FLOAT), false),
            AttributeInfo("value_floats", setOf(AttributeProto.AttributeType.FLOATS), false),
            AttributeInfo("value_int", setOf(AttributeProto.AttributeType.INT), false),
            AttributeInfo("value_ints", setOf(AttributeProto.AttributeType.INTS), false),
            //TODO: sparse tensor values
        )

        private val INPUTS_INFO = emptyList<IOInfo>()

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Constant", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<WebGPUTensor?>, profilingContext: ProfilingContext?): List<WebGPUTensor?> {
        context as WebGPUContext

        //only one of all attributes is not null
        val (name, value) = ATTRIBUTES_INFO.map { it.name to getAttributeOrNull<Any?>(it.name) }.single { it.second != null }

        @Suppress("UNCHECKED_CAST")
        val result = when (name) {
            "value" -> value
            "value_float" -> NDArray(ArrayInfo(intArrayOf(), WebGPUDataType.FLOAT32), data = floatArrayOf(value as Float), device = context.device).asTensor()
            "value_floats" -> {
                value as FloatArray
                NDArray(ArrayInfo(intArrayOf(value.size), WebGPUDataType.FLOAT32), data = value, device = context.device).asTensor()
            }
            "value_int" -> NDArray(ArrayInfo(intArrayOf(), WebGPUDataType.INT32), data = intArrayOf((value as Long).toInt()), device = context.device).asTensor()
            "value_ints" -> {
                value as LongArray
                NDArray(ArrayInfo(intArrayOf(value.size), WebGPUDataType.FLOAT32), data = value.toIntArray(), device = context.device).asTensor()
            }
            else -> error("Unsupported data type")
        } as WebGPUTensor
        result.data.unmap()
        return listOf(result)
    }
}
