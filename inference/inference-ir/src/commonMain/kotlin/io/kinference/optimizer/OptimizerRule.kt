package io.kinference.optimizer

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.protobuf.message.AttributeProto

abstract class OptimizerRule<T : ONNXData<*, *>>(val name: String) {
    companion object {
        const val PREFIX = "KI_Opt"
    }

    abstract fun shouldApply(graph: Graph<T>, operator: Operator<T, T>): Boolean
    abstract suspend fun transform(graph: Graph<T>, operator: Operator<T, T>)

    private suspend fun checkAttributes(operator: Operator<T, T>, report: GraphOptimizer.OptimizationReport) {
        for (attribute in operator.attributes) {
            if (attribute.value.type == AttributeProto.AttributeType.GRAPH)
                apply(attribute.value.value as Graph<T>, report)
            if (attribute.value.type == AttributeProto.AttributeType.GRAPHS)
                for (g in attribute.value.value as List<Graph<T>>) apply(g, report)
        }
    }

    suspend fun apply(graph: Graph<T>, report: GraphOptimizer.OptimizationReport): Graph<T> {
        for (i in graph.operators.indices) {
            val operator = graph.operators[i]
            checkAttributes(operator, report)

            if (shouldApply(graph, operator)) {
                transform(graph, operator)
                report.append(this)
            }
        }
        return graph
    }
}
