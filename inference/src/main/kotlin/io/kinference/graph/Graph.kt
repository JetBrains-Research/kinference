package io.kinference.graph

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.onnx.*
import io.kinference.operators.*
import io.kinference.types.ValueInfo
import org.slf4j.LoggerFactory
import java.util.*

//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
class Graph(proto: GraphProto) {
    companion object {
        private val logger = LoggerFactory.getLogger(Graph::class.java)
    }

    val operators: List<Operator<ONNXData, ONNXData>>
    val inputs = proto.input.map { ValueInfo.create(it) }
    val outputs = proto.output.map { ValueInfo.create(it) }
    val info = proto.value_info.map { ValueInfo.create(it) }
    private val valueOrderInfo = GraphValueOrderInfo()

    val initializers: List<Tensor>
    private val initNames = proto.initializer.map { it.name }
    private val dividerByName: Map<String, Int>

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

        var order = 0
        val outputNames = outputs.map { it.name }
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
                    valueOrderInfo.putOrderFor(node.dependencies - outputNames, order)
                    order++
                }
            } else stack.pop()
        }

        require(operators.size == proto.node.size)

        dividerByName = HashMap<String, Int>().apply {
            for (operator in operators) {
                //TODO: Make normal divider init
                if (operator.info.name == "Attention" || operator.info.name == "QAttention") {
                    val numHeads = (operator.attributes["num_heads"]!!.value as Long).toInt()
                    for (input in operator.info.inputs) {
                        if (input.index in operator.inputs.indices) {
                            val name = operator.inputs[input.index]
                            if (input.name == "weight" || input.name == "bias")
                                put(name, input.divider * numHeads)
                            else
                                put(name, input.divider)
                        }
                    }
                } else {
                    for (input in operator.info.inputs) {
                        if (input.index in operator.inputs.indices) {
                            val name = operator.inputs[input.index]
                            put(name, input.divider)
                        }
                    }
                }
            }
        }

        initializers = proto.initializer.map {
            val divider = dividerByName[it.name] ?: 1

            Tensor.create(it, divider)
        }
    }

    private fun GraphValueOrderInfo.putOrderFor(names: Set<String>, order: Int) {
        val (_, otherNames) = names.partition { name -> initNames.any { it == name } }
        putOrder(otherNames, order)
    }

    val availableInputs: List<String>
        get() = inputs.map { it.name }

    fun prepareInput(name: String, value: List<Any>): Tensor {
        val inputInfo = inputs.find { it.name == name }
        require(inputInfo != null) { "Input with name $name is not found" }
        require(inputInfo is ValueInfo.TensorInfo) { "Only tensor inputs are supported" }

        return Tensor(value, inputInfo.type)
    }

    fun prepareInput(proto: TensorProto): Tensor {
        val divider = dividerByName[proto.name] ?: 1

        return Tensor.create(proto, divider)
    }

    fun prepareInput(value: List<Any>): Tensor {
        require(inputs.size == 1) { "Multiple input nodes found. Specify input name explicitly" }
        val name = inputs.single().name
        return prepareInput(name, value)
    }

    private fun Context.cleanupUntilOrder(order: Int) {
        return this.removeValues { valueOrderInfo.getOrder(it) <= order }
    }

    fun execute(inputs: List<ONNXData>, root: Context? = null): List<ONNXData> {
        //TODO: check that all inputs were set and not null

        val context = Context(root)
        for (tensor in initializers) {
            context.putValue(tensor.info.name, tensor)
        }
        for (input in inputs) {
            if (input.info.name !in availableInputs) {
                logger.warn("Input node '${input.info.name}' not found in Graph and probably is excessive")
                continue
            }
            context.putValue(input.info.name, input)
        }

        //println("\nExec model:")
        for ((i, operator) in operators.withIndex()) {
//            lateinit var outputs: List<ONNXData?>
//            val time = measureNanoTime {
//                outputs = operator.applyWithCheck(context, operator.inputs.map { input -> if (input.isEmpty()) null else context.getValue(input) })
//            }
//
//            println("${operator.info.name} - ${time / 1000000f} ms")

            val outputs = operator.applyWithCheck(context, operator.inputs.map { input -> if (input.isEmpty()) null else context.getValue(input) })

            context.cleanupUntilOrder(i)
            outputs.zip(operator.outputs) { output, variable ->
                if (output == null) require(variable.isEmpty()) { "Required output '$variable' not provided by '${operator.info.name}' operator" }
                if (variable.isNotEmpty()) {
                    context.putValue(variable, output!!.rename(name = variable))
                }
            }
        }
        return outputs.map { context.getValue(it.name) }
    }
}
