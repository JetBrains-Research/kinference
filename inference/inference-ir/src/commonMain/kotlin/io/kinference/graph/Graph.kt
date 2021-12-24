package io.kinference.graph

import io.kinference.data.ONNXData
import io.kinference.model.ExecutionContext
import io.kinference.operator.*
import io.kinference.profiler.profile
import io.kinference.protobuf.message.*
import io.kinference.types.ValueInfo
import io.kinference.utils.LoggerFactory
import io.kinference.utils.Stack
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.time.ExperimentalTime

//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
@ExperimentalTime
abstract class Graph<T : ONNXData<*, *>>(proto: GraphProto, opSetRegistry: OperatorSetRegistry, factory: OperatorFactory<T>) {
    companion object {
        private val logger = LoggerFactory.create("io.kinference.core.graph.Graph")
    }

    val operators: List<Operator<T, T>>
    val inputs = proto.input.map { ValueInfo.create(it) }
    val outputs = proto.output.map { ValueInfo.create(it) }
    val info = proto.valueInfo.map { ValueInfo.create(it) }
    private val valueOrderInfo = GraphValueOrderInfo()

    val initializers: List<T>
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
                    operators.add(factory.create(node.proto, opSetRegistry))
                    valueOrderInfo.putOrderFor(node.dependencies - outputNames, order)
                    order++
                }
            } else stack.pop()
        }

        if (operators.size != proto.node.size) {
            logger.warning {
                "Count of used operators ${operators.size} not equals count of operators in model ${proto.node.size}. " +
                    "Remove unused operators from model for more performance!"
            }
        }

        initializers = proto.initializer.map { prepareInput(it) }
    }

    abstract fun prepareInput(proto: TensorProto): T

    private fun GraphValueOrderInfo.putOrderFor(names: Set<String>, order: Int) {
        val (_, otherNames) = names.partition { name -> initNames.any { it == name } }
        putOrder(otherNames, order)
    }

    val availableInputs: List<String>
        get() = inputs.map { it.name }

    private fun GraphContext<T>.cleanupUntilOrder(order: Int) {
        return this.removeValues { valueOrderInfo.getOrder(it) <= order }
    }

    protected abstract fun makeContext(root: GraphContext<T>?): GraphContext<T>

    @ExperimentalTime
    fun execute(inputs: List<T>, _contexts: Contexts<T> = emptyContexts()): List<T> {
        //TODO: check that all inputs were set and not null
        val contexts = Contexts(makeContext(_contexts.graph), _contexts.profiling, _contexts.execution ?: ExecutionContext(EmptyCoroutineContext))

        for (tensor in initializers) {
            contexts.graph!!.putValue(tensor.name!!, tensor)
        }
        for (input in inputs) {
            if (input.name !in availableInputs) {
                logger.warning { "Input node '${input.name}' not found in Graph and probably is excessive" }
                continue
            }
            contexts.graph!!.putValue(input.name!!, input)
        }

        for ((i, operator) in operators.withIndex()) {
            contexts.execution?.checkCancelled?.invoke()
            
            lateinit var outputs: List<T?>
            contexts.profiling.profile(operator.info.name) { profilingContext ->
                outputs = operator.applyWithCheck(
                    Contexts(contexts.graph, profilingContext, contexts.execution),
                    operator.inputs.map { input -> if (input.isEmpty()) null else contexts.graph!!.getValue(input) })
            }

            contexts.profiling.profile("${operator.info.name}:cleanup") {
                contexts.graph!!.cleanupUntilOrder(i)
                outputs.zip(operator.outputs) { output, variable ->
                    if (output == null) require(variable.isEmpty()) { "Required output '$variable' not provided by '${operator.info.name}' operator" }
                    if (variable.isNotEmpty()) {
                        contexts.graph.putValue(variable, output!!.rename(name = variable) as T)
                    }
                }
            }
        }
        return outputs.map { contexts.graph!!.getValue(it.name) }
    }
}
