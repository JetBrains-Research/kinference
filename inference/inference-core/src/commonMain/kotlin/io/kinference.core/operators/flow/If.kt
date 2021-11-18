package io.kinference.core.operators.flow

import io.kinference.core.KIONNXData
import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIContext
import io.kinference.core.graph.KIGraph
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.LongNDArray
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

    private fun inner(
        context: Context<KIONNXData<*>>,
        body: KIGraph,
        counter: Long,
        condition: Boolean,
        modified: MutableList<KITensor>,
        scans: List<MutableList<KITensor>>
    ): Boolean {
        val inputs = ArrayList<KIONNXData<*>>().apply {
            add(LongNDArray.scalar(counter).asTensor(body.inputs[0].name))
            add(BooleanNDArray.scalar(condition).asTensor(body.inputs[1].name))
            body.inputs.drop(2).zip(modified) { input, value ->
                add(value.rename(input.name))
            }
        }

        val outputs = body.execute(inputs, context)
        val iterationOutputs = outputs.drop(body.inputs.size - 1)

        modified.clear()
        // remove keepgoing flag (first) and take only modified (without scans)
        for (output in outputs.drop(1).take(body.inputs.size - 2)) {
            modified.add(output as KITensor)
        }

        require(iterationOutputs.size == scans.size) { "Loop subgraph didn't provide expected output count" }
        scans.zip(iterationOutputs) { buffer, output ->
            buffer.add(output as KITensor)
        }

        return (outputs[0] as KITensor).data.singleValue() as Boolean
    }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val condition = inputs[0]!!.data.singleValue() as Boolean
        val outputs = if (condition) thenBranch.execute(emptyList(), context as KIContext, profilingContext) else elseBranch.execute(emptyList(), context as KIContext, profilingContext)

        return outputs as List<KITensor>
    }
}
