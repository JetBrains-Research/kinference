package io.kinference.core.graph

import io.kinference.core.data.tensor.KITensor
import io.kinference.core.operators.Operator
import io.kinference.data.ONNXData
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
abstract class ContextPrepare {
    abstract fun appendContext(context: Context, initializers: List<KITensor>, operator: Operator<ONNXData<*>, ONNXData<*>>)

    @OptIn(ExperimentalTime::class)
    protected fun initTensorByDefaultName(defaultName: String, operator: Operator<ONNXData<*>, ONNXData<*>>, initializers: List<KITensor>): KITensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.name == tensorName }
    }
}
