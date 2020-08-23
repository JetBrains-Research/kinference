package org.jetbrains.research.kotlin.inference.graph

import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.onnx.GraphProto
import org.jetbrains.research.kotlin.inference.onnx.NodeProto
import org.jetbrains.research.kotlin.inference.operators.Operator
import org.jetbrains.research.kotlin.inference.operators.OperatorFactory
import org.jetbrains.research.kotlin.inference.types.ValueInfo
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet

//TODO: support general graphs
//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
@ExperimentalUnsignedTypes
class Graph(proto: GraphProto) {
    val operators: List<Operator<ONNXData, ONNXData>>
    val inputs = proto.input.map { ValueInfo.create(it) }
    val outputs = proto.output.map { ValueInfo.create(it) }
    val info = proto.value_info.map { ValueInfo.create(it) }

    val initializers = proto.initializer.map { Tensor.create(it) }

    private data class Node(val proto: NodeProto, var visited: Boolean = false) {
        private fun NodeProto.collectRequiredInputs(): Set<String> = HashSet<String>().apply {
            for (variable in input) {
                if (variable.isNotEmpty()) add(variable)
            }

            for (attr in attribute) {
                if (attr.type == AttributeProto.AttributeType.GRAPH) {
                    val subGraphInputs: HashSet<String> = attr.g!!.input.mapTo(HashSet()) { it.name!! }

                    val subGraphLocalInputs = attr.g.node.flatMapTo(HashSet()) { it.collectRequiredInputs() }
                    subGraphInputs.addAll(attr.g.output.map { it.name!! })

                    val subGraphLocalOutputs = attr.g.node.flatMapTo(HashSet()) { it.output }

                    addAll((subGraphLocalInputs - subGraphLocalOutputs) - subGraphInputs)
                }
                // TODO AttributeProto.AttributeType.GRAPHS
            }
        }

        val dependencies by lazy { proto.collectRequiredInputs() }
    }

    init {
        operators = ArrayList(proto.node.size)
        val nodes = HashMap<String, Node>().apply {
            for (nodeProto in proto.node) {
                val node = Node(nodeProto)
                for (output in nodeProto.output) {
                    put(output, node)
                }
            }
        }

        val stack = Stack<Node>().apply {
            for (output in proto.output) {
                val name = output.name!!
                if (name.isNotEmpty()) {
                    val node = nodes[name]
                    if (node != null) push(node)
                }
            }
        }

        while (stack.isNotEmpty()) {
            val node = stack.peek()
            if (!node.visited) {
                var ready = true
                for (input in node.dependencies) {
                    val next = nodes[input]
                    if (next != null && !next.visited) {
                        ready = false
                        stack.push(next)
                    }
                }

                if (ready) {
                    node.visited = true
                    stack.pop()
                    operators.add(OperatorFactory.create(node.proto))
                }
            } else stack.pop()
        }

        require(operators.size == proto.node.size)
    }

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
                    context.putValue(variable, output!!.rename(newName = variable))
                }
            }
        }

        return outputs.map { context.getValue(it.name) }
    }
}
