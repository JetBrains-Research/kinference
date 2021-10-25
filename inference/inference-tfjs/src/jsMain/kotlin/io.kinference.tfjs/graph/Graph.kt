package io.kinference.tfjs.graph

import io.kinference.protobuf.message.*
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.operators.*
import io.kinference.tfjs.types.ValueInfo
import io.kinference.tfjs.utils.Stack
import io.kinference.utils.LoggerFactory

class Graph(proto: GraphProto) {
    companion object {
        private val logger = LoggerFactory.create("TFJS Graph") //logger(Graph::class.simpleName ?: "")
    }

    val operators: List<Operator<TFJSData<*>, TFJSData<*>>>
    val inputs = proto.input.map { ValueInfo.create(it) }
    val outputs = proto.output.map { ValueInfo.create(it) }
    val info = proto.valueInfo.map { ValueInfo.create(it) }
    private val valueOrderInfo = GraphValueOrderInfo()

    val initializers: List<TFJSTensor>
    private val initNames = proto.initializer.map { it.name }

    private data class Node(val proto: NodeProto, var visited: Boolean = false) {
        private fun NodeProto.collectRequiredInputs(): Set<String> = HashSet<String>().apply {
            for (variable in input) {
                if (variable.isNotEmpty()) add(variable)
            }

            for (attr in attribute) {
                if (attr.type == AttributeProto.AttributeType.GRAPH) {
                    val subGraphInputs: HashSet<String> = attr.g!!.input.mapTo(HashSet()) { it.name!! }

                    val subGraphLocalInputs = attr.g!!.node.flatMapTo(HashSet()) { it.collectRequiredInputs() }
                    subGraphInputs.addAll(attr.g!!.output.map { it.name!! })

                    val subGraphLocalOutputs = attr.g!!.node.flatMapTo(HashSet()) { it.output }

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

        initializers = proto.initializer.map { TFJSTensor.create(it) }
    }

    private fun GraphValueOrderInfo.putOrderFor(names: Set<String>, order: Int) {
        val (_, otherNames) = names.partition { name -> initNames.any { it == name } }
        putOrder(otherNames, order)
    }

    val availableInputs = inputs.map { it.name }

    fun prepareInput(proto: TensorProto) = TFJSTensor.create(proto)

    private fun Context.cleanupUntilOrder(order: Int) {
        return this.removeValues { it !in initNames && it !in availableInputs && valueOrderInfo.getOrder(it) <= order }
    }

    fun execute(inputs: Collection<TFJSData<*>>, root: Context? = null): List<TFJSData<*>> {
        //TODO: check that all inputs were set and not null

        val context = Context(root)
        for (tensor in initializers) {
            context.putValue(tensor.name ?: "", tensor)
        }
        for (input in inputs) {
            if (input.name !in availableInputs) {
                logger.warning { "Input node '${input.name}' not found in Graph and probably is excessive" }
                continue
            }
            context.putValue(input.name ?: "", input)
        }

        for ((i, operator) in operators.withIndex()) {
            val outputs = operator.applyWithCheck(context, operator.inputs.map { input -> if (input.isEmpty()) null else context.getValue(input) })

//            profilingContext.profile("${operator.info.name}:cleanup") {
                context.cleanupUntilOrder(i)
                outputs.zip(operator.outputs) { output, variable ->
                    if (output == null) require(variable.isEmpty()) { "Required output '$variable' not provided by '${operator.info.name}' operator" }
                    if (variable.isNotEmpty()) {
                        context.putValue(variable, output!!.rename(name = variable))
                    }
                }
//            }
        }
        return outputs.map { context.getValue(it.name) }
    }
}
