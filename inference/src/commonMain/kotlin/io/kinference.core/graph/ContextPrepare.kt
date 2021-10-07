package io.kinference.core.graph

import io.kinference.core.data.KIONNXData
import io.kinference.core.data.tensors.KITensor
import io.kinference.core.operators.Operator
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
abstract class ContextPrepare {
    abstract fun appendContext(context: Context, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>)

    @OptIn(ExperimentalTime::class)
    protected fun initTensorByDefaultName(defaultName: String, operator: Operator<KIONNXData<*>, KIONNXData<*>>, initializers: List<KITensor>): KITensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.info.name == tensorName }
    }
}
