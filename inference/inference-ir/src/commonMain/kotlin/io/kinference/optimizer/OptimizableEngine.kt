package io.kinference.optimizer

import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.model.Model

enum class OptLevel(val level: Int) {
    NO_OPT(0), DEFAULT(1), ALL(2)
}

interface OptimizableEngine<T : ONNXData<*, *>> : InferenceEngine<T> {
    fun optimizeModel(model: Model<T>, rules: List<OptimizerRule<T>>): Model<T>
    fun optimizeModel(model: Model<T>, level: OptLevel): Model<T>
}
