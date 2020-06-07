package org.jetbrains.research.kotlin.mpp.inference.operators

import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.*
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm.LSTMFactory
import org.jetbrains.research.kotlin.mpp.inference.operators.math.Add
import org.jetbrains.research.kotlin.mpp.inference.operators.math.MatMul
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass

object OperatorFactory {
    private inline fun <reified T : Number> create(name: String?, attributes: Map<String, Attribute<*>>): Operator<T> = when (name) {
        "Add" -> Add()
        "MatMul" -> MatMul()
        "Identity" -> Identity()
        "Relu" -> Relu()
        "Sigmoid" -> Sigmoid()
        "Tanh" -> Tanh()
        "Softmax" -> Softmax(attributes["axis"]?.value as? Long)
        "LSTM" -> LSTMFactory.create(attributes)
        else -> error("Unsupported operator")
    }

    @Suppress("UNCHECKED_CAST")
    fun create(type: DataType?, name: String?, attributes: Map<String, Attribute<*>>): Operator<Number> = when (type?.resolveKClass()!!) {
        Float::class -> create<Float>(name, attributes)
        Double::class -> create<Double>(name, attributes)
        Long::class -> create<Long>(name, attributes)
        Int::class -> create<Int>(name, attributes)
        else -> error("Unsupported data type")
    } as Operator<Number>
}
