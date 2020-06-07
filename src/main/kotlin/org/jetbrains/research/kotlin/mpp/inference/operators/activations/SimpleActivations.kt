package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import kotlin.math.exp

class Identity<T : Number> : Activation<T>() {
    override fun activate(input: Tensor<T>): Tensor<T> = input
}

class Relu<T : Number> : Activation<T>() {
    override fun activate(input: Tensor<T>): Tensor<T> = input.mapElements { x -> max(0, x) }
}

//only for float and double types
class Sigmoid<T : Number> : Activation<T>() {
    override fun activate(input: Tensor<T>): Tensor<T> = input.mapElements { x ->
        (1.0 / (1.0 + exp(-x.toDouble()))) as T
    }
}

//only for float and double types
class Tanh<T : Number> : Activation<T>() {
    override fun activate(input: Tensor<T>): Tensor<T> = input.mapElements { x ->
        ((exp(2.0 * x.toDouble()) - 1.0) / (exp(2.0 * x.toDouble()) + 1.0)) as T
    }
}
