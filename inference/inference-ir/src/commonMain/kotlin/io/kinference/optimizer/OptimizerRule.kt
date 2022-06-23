package io.kinference.optimizer

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.protobuf.message.AttributeProto

abstract class OptimizerRule<T : ONNXData<*, *>>(val name: String, val type: RuleType) {
    enum class RuleType {
        MODIFY,
        REMOVE,
        MERGE
    }

    companion object {
        const val PREFIX = "KI_Opt"
    }

    abstract fun shouldApply(graph: Graph<T>, name: String): Boolean
    abstract fun transform(graph: Graph<T>, name: String)

    private fun checkAttributes(graph: Graph<T>, name: String, report: GraphOptimizer.OptimizationReport) {
        val operator = graph.operators.singleOrNull { it.name == name } ?: return
        for (attribute in operator.attributes) {
            if (attribute.value.type == AttributeProto.AttributeType.GRAPH)
                apply(attribute as Graph<T>, report)
            if (attribute.value.type == AttributeProto.AttributeType.GRAPHS)
                for (g in attribute.value as List<Graph<T>>) apply(g, report)
        }
    }

    fun apply(graph: Graph<T>, report: GraphOptimizer.OptimizationReport): Graph<T> {
        var pos = 0
        while (pos < graph.operators.size) {
            val operator = graph.operators[pos]
            checkAttributes(graph, operator.name, report)

            if (operator.name.isEmpty()) {
                pos++
                continue
            }

            if (shouldApply(graph, operator.name)) {
                transform(graph, operator.name)
                report.append(this)
                if (type != RuleType.MODIFY) continue
            }
            pos++
        }
        return graph
    }
}
