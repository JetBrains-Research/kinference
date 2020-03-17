package org.jetbrains.research.kotlin.mpp.inference.operators.math

import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class MatMul<T : Number> : Operator<T>() {
    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        require(inputs.size == 2) { "Applicable only for two arguments" }

        val left = inputs.first()
        val right = inputs.last()

        return listOf(left dot right)
    }
}
