package io.kinference.tfjs.graph

import io.kinference.graph.GraphContext
import io.kinference.operator.Operator
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor

abstract class ContextPrepare {
    abstract suspend fun appendContext(context: GraphContext<TFJSData<*>>, initializers: List<TFJSTensor>, operator: Operator<TFJSData<*>, TFJSData<*>>)

    protected fun initTensorByDefaultName(defaultName: String, operator: Operator<TFJSData<*>, TFJSData<*>>, initializers: List<TFJSTensor>): TFJSTensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.name == tensorName }
    }
}
