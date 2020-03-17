package org.jetbrains.research.kotlin.mpp.inference.operators

import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

@Suppress("UNCHECKED_CAST")
abstract class Operator<T : Number> {
    abstract fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>>

    companion object {
        operator fun invoke(name: String?, type: DataType?, value: Collection<Tensor<*>>): Collection<Tensor<*>> {
            return OperatorFactory.create(type, name).apply(value as Collection<Tensor<Number>>)
        }
    }
}
