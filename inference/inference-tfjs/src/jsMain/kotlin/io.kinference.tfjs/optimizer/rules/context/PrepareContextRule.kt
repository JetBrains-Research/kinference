package io.kinference.tfjs.optimizer.rules.context

import io.kinference.operator.Operator
import io.kinference.optimizer.OptimizerRule
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor

abstract class PrepareContextRule(operatorName: String) : OptimizerRule<TFJSData<*>>(name = "Optimize $operatorName context") {
    protected fun initTensorByDefaultName(defaultName: String, operator: Operator<TFJSData<*>, TFJSData<*>>, initializers: List<TFJSTensor>): TFJSTensor? {
        val index = operator.info.inputs.find { it.name == defaultName }?.index ?: return null
        val tensorName = operator.inputs.getOrNull(index)

        return initializers.find { it.name == tensorName }
    }
}
