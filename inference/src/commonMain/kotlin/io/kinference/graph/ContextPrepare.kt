package io.kinference.graph

import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.operators.Operator
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
abstract class ContextPrepare {
    abstract fun appendContext(context: Context, initializers: List<Tensor>, operator: Operator<ONNXData, ONNXData>)


    @OptIn(ExperimentalTime::class)
    protected fun initTensorByDefaultName(defaultName: String, operator: Operator<ONNXData, ONNXData>, initializers: List<Tensor>): Tensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.info.name == tensorName }
    }
}
