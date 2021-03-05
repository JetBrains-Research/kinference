package io.kinference.operators.flow

import io.kinference.attributes.Attribute
import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.Graph
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.operators.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

class If(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
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

        private val INFO = OperatorInfo("If", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val thenBranch: Graph by attribute("then_branch")
    private val elseBranch: Graph by attribute("else_branch")

    private fun inner(
        context: Context,
        body: Graph,
        counter: Long,
        condition: Boolean,
        modified: MutableList<Tensor>,
        scans: List<MutableList<Tensor>>
    ): Boolean {
        val inputs = ArrayList<ONNXData>().apply {
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
            modified.add(output as Tensor)
        }

        require(iterationOutputs.size == scans.size) { "Loop subgraph didn't provide expected output count" }
        scans.zip(iterationOutputs) { buffer, output ->
            buffer.add(output as Tensor)
        }

        return (outputs[0] as Tensor).data.singleValue() as Boolean
    }

    override fun apply(context: Context, inputs: List<Tensor?>): List<Tensor?> {
        val condition = inputs[0]!!.data.singleValue() as Boolean
        val outputs = if (condition) thenBranch.execute(emptyList(), context) else elseBranch.execute(emptyList(), context)

        return outputs as List<Tensor>
    }
}
