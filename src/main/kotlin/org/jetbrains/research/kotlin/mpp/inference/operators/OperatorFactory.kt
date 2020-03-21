package org.jetbrains.research.kotlin.mpp.inference.operators

import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Activation
import org.jetbrains.research.kotlin.mpp.inference.operators.math.Add
import org.jetbrains.research.kotlin.mpp.inference.operators.math.MatMul
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass

object OperatorFactory {
    private inline fun <reified T : Number> create(name: String?): Operator<T> = when (name) {
        "Add" -> Add()
        "MatMul" -> MatMul()
        "Identity" -> Activation.Identity()
        "Relu" -> Activation.Relu()
        "Sigmoid" -> Activation.Sigmoid()
        else -> error("Unsupported operator")
    }

    @Suppress("UNCHECKED_CAST")
    fun create(type: DataType?, name: String?): Operator<Number> = when (type?.resolveKClass()!!) {
        Float::class -> create<Float>(name)
        Double::class -> create<Double>(name)
        Long::class -> create<Long>(name)
        Int::class -> create<Int>(name)
        else -> error("Unsupported data type")
    } as Operator<Number>
}
