package org.jetbrains.research.kotlin.mpp.inference.operators.math

import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class Add<T : Number> : Operator<T>() {
    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        return listOf(inputs.first() + inputs.last())
    }
}
