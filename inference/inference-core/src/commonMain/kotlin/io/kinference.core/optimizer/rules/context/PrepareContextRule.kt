package io.kinference.core.optimizer.rules.context

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.operator.Operator
import io.kinference.optimizer.OptimizerRule

abstract class PrepareContextRule(operatorName: String) : OptimizerRule<KIONNXData<*>>(name = "Optimize $operatorName context") {
   // abstract suspend fun appendContext(context: GraphContext<KIONNXData<*>>, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>)

    protected fun initTensorByDefaultName(defaultName: String, operator: Operator<KIONNXData<*>, KIONNXData<*>>, initializers: List<KITensor>): KITensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.name == tensorName }
    }
}
