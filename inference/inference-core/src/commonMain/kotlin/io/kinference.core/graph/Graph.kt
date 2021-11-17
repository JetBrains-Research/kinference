package io.kinference.core.graph

import io.kinference.core.KIONNXData
import io.kinference.core.utils.Stack
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.model.KIModel
import io.kinference.core.operators.*
import io.kinference.core.operators.layer.attention.AttentionContext
import io.kinference.core.operators.layer.recurrent.gru.GRUContext
import io.kinference.core.operators.layer.recurrent.lstm.LSTMContext
import io.kinference.core.operators.quantization.lstm.DynamicQuantizeLSTMContext
import io.kinference.protobuf.message.*
import io.kinference.core.types.ValueInfo
import io.kinference.profiler.ProfilingContext
import io.kinference.profiler.profile
import io.kinference.utils.LoggerFactory
import kotlin.time.ExperimentalTime

//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
@ExperimentalTime
class Graph(proto: GraphProto, opSetRegistry: KIModel.OperatorSetRegistry) {
    companion object {
        private val logger = LoggerFactory.create("io.kinference.core.graph.Graph")
    }

    val operators: List<Operator<KIONNXData<*>, KIONNXData<*>>>
    val inputs = proto.input.map { ValueInfo.create(it) }
    val outputs = proto.output.map { ValueInfo.create(it) }
    val info = proto.valueInfo.map { ValueInfo.create(it) }
    private val valueOrderInfo = GraphValueOrderInfo()

    val initializers: List<KITensor>
    private val initNames = proto.initializer.map { it.name }
    private val preparedTensorsContext = Context()

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
                    operators.add(OperatorFactory.create(node.proto, opSetRegistry))
                    valueOrderInfo.putOrderFor(node.dependencies - outputNames, order)
                    order++
                }
            } else stack.pop()
        }

        if (operators.size != proto.node.size) {
            logger.warning { "Count of used operators ${operators.size} not equals count of operators in model ${proto.node.size}. " +
                             "Remove unused operators from model for more performance!" }
        }
//        require(operators.size == proto.node.size)

        initializers = proto.initializer.map { KITensor.create(it) }

        for (operator in operators) {
            when(operator.info.name) {
                "LSTM" -> LSTMContext.appendContext(preparedTensorsContext, initializers, operator)
                "DynamicQuantizeLSTM" -> DynamicQuantizeLSTMContext.appendContext(preparedTensorsContext, initializers, operator)
                "GRU" -> GRUContext.appendContext(preparedTensorsContext, initializers, operator)
                "Attention", "QAttention" -> AttentionContext.appendContext(preparedTensorsContext, initializers, operator)
            }
        }
    }

    private fun GraphValueOrderInfo.putOrderFor(names: Set<String>, order: Int) {
        val (_, otherNames) = names.partition { name -> initNames.any { it == name } }
        putOrder(otherNames, order)
    }

    val availableInputs: List<String>
        get() = inputs.map { it.name }

    private fun Context.cleanupUntilOrder(order: Int) {
        return this.removeValues { valueOrderInfo.getOrder(it) <= order }
    }

    @ExperimentalTime
    fun execute(inputs: List<KIONNXData<*>>, root: Context? = null, profilingContext: ProfilingContext? = null): List<KIONNXData<*>> {
        //TODO: check that all inputs were set and not null

        val context = Context(root)
        context.mergeContext(preparedTensorsContext)
        for (tensor in initializers) {
            context.putValue(tensor.name!!, tensor)
        }
        for (input in inputs) {
            if (input.name !in availableInputs) {
                logger.warning { "Input node '${input.name}' not found in Graph and probably is excessive" }
                continue
            }
            context.putValue(input.name!!, input)
        }

        for ((i, operator) in operators.withIndex()) {
            lateinit var outputs: List<KIONNXData<*>?>
            profilingContext.profile(operator.info.name) { profilingContext ->
                outputs = operator.applyWithCheck(context, operator.inputs.map { input -> if (input.isEmpty()) null else context.getValue(input) }, profilingContext)
            }

            profilingContext.profile("${operator.info.name}:cleanup") {
                context.cleanupUntilOrder(i)
                outputs.zip(operator.outputs) { output, variable ->
                    if (output == null) require(variable.isEmpty()) { "Required output '$variable' not provided by '${operator.info.name}' operator" }
                    if (variable.isNotEmpty()) {
                        context.putValue(variable, output!!.rename(name = variable))
                    }
                }
            }
        }
        return outputs.map { context.getValue(it.name) }
    }
}
