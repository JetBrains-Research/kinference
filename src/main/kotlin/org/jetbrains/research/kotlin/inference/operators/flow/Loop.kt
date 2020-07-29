package org.jetbrains.research.kotlin.inference.operators.flow

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.tensor.stack
import org.jetbrains.research.kotlin.inference.graph.Graph
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.*

class Loop(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int) : Operator<Tensor, Tensor>(INFO, usedOutputsNum, attributes) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("body", setOf(AttributeProto.AttributeType.GRAPH), required = true)
        )

        private val INPUTS_INFO = listOf(
            // TODO: add "" like in docs as ignore parameter
            InputInfo(0, setOf(TensorProto.DataType.INT64), "M", true),
            InputInfo(1, setOf(TensorProto.DataType.BOOL), "cond", true),
            VariadicInputInfo(2, TYPE_CONSTRAINTS, "v_initial")
        )

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "Y"))

        private val INFO = OperatorInfo("Loop", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    override fun apply(inputs: List<Tensor>): List<Tensor> {
        val maxTripCount = inputs[0].data[0] as Long
        var keepgoing = inputs[1].data[0] as Boolean

        val body = getAttributeValue("body") as Graph
        require(body.inputs.size == inputs.size) { "Not enough inputs for Loop subgraph\nPresent: ${inputs.size}, Expected: ${body.inputs.size}" }
        body.inputs.drop(2).zip(inputs.drop(2)) { input, value ->
            body.setInput(value.clone(input.name))
        }

        // NOTE: works as ONNX Runtime (counter and condition are ignored and not returned to results of Loop)
        val modifiedCount = body.inputs.size - 2
        val modified = inputs.drop(2).toMutableList()

        val scansCount = body.outputs.size - 1 - modifiedCount
        val scans = (0 until scansCount).map { ArrayList<Tensor>() } // TODO support scalar tensors for Loop

        repeat(maxTripCount.toInt()) { counter ->
            if (keepgoing) {
                body.setInput(Tensor(counter.toLong(), TensorProto.DataType.INT64, name = body.inputs[0].name))
                body.setInput(Tensor(keepgoing, TensorProto.DataType.BOOL, name = body.inputs[1].name))

                val outputs = body.execute()
                val iterationOutputs = outputs.drop(body.inputs.size - 1)
                keepgoing = (outputs[0] as Tensor).data[0] as Boolean

                modified.clear()
                body.inputs.drop(2).zip(outputs.drop(1)) { input, value ->
                    body.setInput(value.clone(input.name))
                    modified.add(value as Tensor)
                }

                require(iterationOutputs.size == scans.size) { "Loop subgraph didn't provide expected output count" }
                scans.zip(iterationOutputs) { buffer, output ->
                    buffer.add(output as Tensor)
                }
            }
        }

        return modified + scans.map { it.stack(0) }
    }
}
