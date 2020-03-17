package org.jetbrains.research.kotlin.mpp.inference.operators.math

import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

//TODO: numpy-like multidirectional broadcasting
class Add<T : Number> : Operator<T>() {
    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        require(inputs.size == 2) { "Applicable only for two arguments" }
        return listOf((inputs.first() + inputs.last())!!)
    }
}
