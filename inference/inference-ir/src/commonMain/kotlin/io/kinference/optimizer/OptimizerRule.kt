package io.kinference.optimizer

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.protobuf.message.AttributeProto

abstract class OptimizerRule<T : ONNXData<*, *>>(val name: String) {
    abstract fun shouldApply(graph: Graph<T>, name: String): Boolean
    abstract fun transform(graph: Graph<T>, name: String)

    private fun checkAttributes(graph: Graph<T>, name: String) {
        val operator = graph.operators.singleOrNull() { it.name == name } ?: return
        for (attribute in operator.attributes) {
            if (attribute.value.type == AttributeProto.AttributeType.GRAPH)
                apply(attribute as Graph<T>)
            if (attribute.value.type == AttributeProto.AttributeType.GRAPHS)
                for (g in attribute.value as List<Graph<T>>) apply(g)
        }
    }

    fun apply(graph: Graph<T>): Graph<T> {
        for (operator in graph.operators) {
            checkAttributes(graph, operator.name)
            if (shouldApply(graph, operator.name)) transform(graph, operator.name)
        }
        return graph
    }
}
