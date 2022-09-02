package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.dtype
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.ndarray.arrays.range

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
        private val TYPE_CONSTRAINTS = setOf(DataType.FLOAT, DataType.DOUBLE, DataType.INT16, DataType.INT32, DataType.INT64)

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

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val start = inputs[0]!!.data
            val limit = inputs[1]!!.data
            val delta = inputs[2]!!.data

            require(start.dtype == limit.dtype && limit.dtype == delta.dtype)
                { "Input tensors must have equal dtype, present: start: ${start.dtype}, limit: ${limit.dtype}, delta: ${delta.dtype}" }

            val startNumber = if (start.dtype == "float32") start.dataFloat().first() else start.dataInt().first()
            val limitNumber = if (limit.dtype == "float32") limit.dataFloat().first() else limit.dataInt().first()
            val deltaNumber = if (delta.dtype == "float32") delta.dataFloat().first() else delta.dataInt().first()

            return@tidy arrayOf(range(startNumber, limitNumber, deltaNumber, start.dtype))
        }

        return listOf(outputs[0].asTensor("output"))
    }
}
