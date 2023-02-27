package io.kinference.core.graph

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.graph.GraphContext
import io.kinference.operator.Operator
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
abstract class ContextPrepare {
    abstract suspend fun appendContext(context: GraphContext<KIONNXData<*>>, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>)

    @OptIn(ExperimentalTime::class)
    protected fun initTensorByDefaultName(defaultName: String, operator: Operator<KIONNXData<*>, KIONNXData<*>>, initializers: List<KITensor>): KITensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.name == tensorName }
    }
}
