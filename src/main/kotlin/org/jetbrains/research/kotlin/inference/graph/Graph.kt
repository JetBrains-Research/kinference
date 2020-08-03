package org.jetbrains.research.kotlin.inference.graph

import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.onnx.GraphProto
import org.jetbrains.research.kotlin.inference.operators.OperatorFactory
import org.jetbrains.research.kotlin.inference.types.ValueInfo

//TODO: support general graphs
//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
class Graph(proto: GraphProto) {
    val operators = proto.node.map { OperatorFactory.create(it) }
    val inputs = proto.input.map { ValueInfo.create(it) }
    val outputs = proto.output.map { ValueInfo.create(it) }
    val info = proto.value_info.map { ValueInfo.create(it) }

    val initializers = proto.initializer.map { Tensor.create(it) }

    val availableInputs: List<String>
        get() = inputs.map { it.name }

    fun prepareInput(name: String, value: List<Any>): Tensor {
        val type = inputs.find { it.name == name }?.type!!
        return Tensor(value, type)
    }

    fun prepareInput(value: List<Any>): Tensor {
        require(inputs.size == 1) { "Multiple input nodes found. Specify input name explicitly" }
        val name = inputs.single().name
        return prepareInput(name, value)
    }

    //only for sequential models
    fun execute(inputs: List<ONNXData>, root: Context? = null): List<ONNXData> {
        //TODO: check that all inputs were set and not null

        val context = Context(root)
        for (tensor in initializers) {
            context.putValue(tensor.info.name, tensor)
        }

        for (input in inputs) {
            require(input.info.name in availableInputs) { "Input node '${input.info.name}' not found in Graph" }
            context.putValue(input.info.name, input)
        }

        for (operator in operators) {
            val outputs = operator.applyWithCheck(context, operator.inputs.map { input -> if (input.isEmpty()) null else context.getValue(input) })
            outputs.zip(operator.outputs) { output, variable ->
                if (output == null) require(variable.isEmpty()) { "Required output '$variable' not provided by '${operator.info.name}' operator" }
                if (variable.isNotEmpty()) {
                    context.putValue(variable, output!!.clone(newName = variable))
                }
            }
        }

        return outputs.map { context.getValue(it.name) }
    }
}
