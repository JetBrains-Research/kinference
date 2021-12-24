package io.kinference.core.operators.flow

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.graph.KIGraph
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class If(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in IfVer1.VERSION.asRange() -> IfVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of If operator: $version")
        }
    }
}

@ExperimentalTime
class IfVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : If(INFO, attributes, inputs, outputs) {
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

    private val thenBranch: KIGraph by attribute("then_branch")
    private val elseBranch: KIGraph by attribute("else_branch")

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val condition = inputs[0]!!.data.singleValue() as Boolean
        val outputs = if (condition) thenBranch.execute(emptyList(), contexts as Contexts<KIONNXData<*>>) else elseBranch.execute(emptyList(), contexts as Contexts<KIONNXData<*>>)

        return outputs as List<KITensor>
    }
}
