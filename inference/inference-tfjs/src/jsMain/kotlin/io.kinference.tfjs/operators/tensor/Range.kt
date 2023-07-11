package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.primitives.types.DataType

sealed class Range(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in RangeVer11.VERSION.asRange() -> RangeVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Constant operator: $version")
            }
    }
}

class RangeVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Range(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.INT16,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64
        )

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "start", optional = false, differentiable = false),
            IOInfo(1, TYPE_CONSTRAINTS, "limit", optional = false, differentiable = false),
            IOInfo(2, TYPE_CONSTRAINTS, "delta", optional = false, differentiable = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false, differentiable = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("Range", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidyNDArray {
            val start = inputs[0]!!.data as NumberNDArrayTFJS
            val limit = inputs[1]!!.data as NumberNDArrayTFJS
            val delta = inputs[2]!!.data as NumberNDArrayTFJS

            require(start.type == limit.type && limit.type == delta.type)
            { "Input tensors must have equal dtype, present: start: ${start.type}, limit: ${limit.type}, delta: ${delta.type}" }


            val startNumber = start.singleValue()
            val limitNumber = limit.singleValue()
            val deltaNumber = delta.singleValue()

            return@tidyNDArray when (start.type) {
                DataType.FLOAT -> NDArrayTFJS.floatRange(startNumber, limitNumber, deltaNumber)
                DataType.INT -> NDArrayTFJS.intRange(startNumber, limitNumber, deltaNumber)
                else -> error("Unsupported data type")
            }
        }

        return listOf(outputs.asTensor("output"))
    }
}
