package org.jetbrains.research.kotlin.mpp.inference.operators

import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

@Suppress("UNCHECKED_CAST")
abstract class Operator<T : Number> {
    abstract fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>>
    open fun apply(vararg inputs: Tensor<T>): Collection<Tensor<T>> = apply(inputs.toList())

    companion object {
        operator fun invoke(name: String?, type: DataType?, value: Collection<Tensor<*>>, attributes: Map<String, Attribute<*>>): Collection<Tensor<*>> {
            return OperatorFactory.create(type, name, attributes).apply(value as Collection<Tensor<Number>>)
        }
    }
}
