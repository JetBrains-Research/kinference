package io.kinference.tfjs.operators.flow

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.graph.TFJSGraph

sealed class If(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): If {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in IfVer1.VERSION.asRange() -> IfVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of If operator: $version")
            }
        }
    }
}


class IfVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : If(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("then_branch", setOf(AttributeProto.AttributeType.GRAPH), required = true),
            AttributeInfo("else_branch", setOf(AttributeProto.AttributeType.GRAPH), required = true)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "cond", optional = false, scalar = true)
        )

        private val OUTPUTS_INFO = listOf(VariadicIOInfo(0, TYPE_CONSTRAINTS, "outputs", minimumArity = 1))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("If", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val thenBranch: TFJSGraph by attribute("then_branch")
    private val elseBranch: TFJSGraph by attribute("else_branch")

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val condition = inputs[0]!!.data.singleValue() as Boolean

        contexts as Contexts<TFJSData<*>>
        val outputs = if (condition) {
            thenBranch.execute(emptyList(), contexts)
        } else {
            elseBranch.execute(emptyList(), contexts)
        }

        return outputs as List<TFJSTensor?>
    }
}
