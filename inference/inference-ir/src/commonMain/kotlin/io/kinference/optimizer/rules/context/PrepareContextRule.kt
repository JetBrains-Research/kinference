package io.kinference.optimizer.rules.context

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.optimizer.OptimizerRule

abstract class PrepareContextRule<T : ONNXData<*, *>>(operatorName: String) : OptimizerRule<T>(name = "Optimize $operatorName context") {
    protected fun <V : T> initTensorByDefaultName(defaultName: String, operator: Operator<T, T>, initializers: List<V>): V? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.name == tensorName }
    }

    protected suspend fun tryRemoveDefaultInitializer(graph: Graph<T>, name: String) {
        val numUsages = graph.countNumberOfInputUsages(name)
        if (numUsages != 1) return

        graph.removeInitializer(name)
    }
}
