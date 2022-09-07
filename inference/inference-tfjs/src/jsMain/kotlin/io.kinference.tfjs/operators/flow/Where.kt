package io.kinference.tfjs.operators.flow

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.*

sealed class Where(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in WhereVer1.VERSION.asRange() -> WhereVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Where operator: $version")
            }
    }
}

class WhereVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Where(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = emptyList<AttributeInfo>()

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "condition", optional = false),
            IOInfo(1, ALL_DATA_TYPES, "X", optional = false),
            IOInfo(2, ALL_DATA_TYPES, "Y", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("Where", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidyNDArrays {
            val condition = inputs[0]!!.data
            val left = inputs[1]!!.data
            val right = inputs[2]!!.data
            return@tidyNDArrays arrayOf(left.where(condition, right))
        }

        return outputs.asNamedOutputs(this.outputs)
    }
}

