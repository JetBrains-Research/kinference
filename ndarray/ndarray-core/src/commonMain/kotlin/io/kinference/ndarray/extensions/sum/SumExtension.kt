package io.kinference.ndarray.extensions.sum

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun List<NumberNDArrayCore>.sum(): NumberNDArrayCore {
    if (isEmpty()) error("Array for sum operation must have at least one element")
    if (size == 1) return single()

    val inputType = this.first().type
    require(this.all { it.type == inputType }) { "Input tensors must have the same data type" }

    return when (inputType) {
        DataType.DOUBLE -> (this as List<DoubleNDArray>).sum()
        DataType.FLOAT -> (this as List<FloatNDArray>).sum()
        else -> error("Unsupported data type in sum operation, tensors must have Float or Double data type, current is $inputType")
    }
}

suspend fun Array<out NumberNDArrayCore>.sum() = this.toList().sum()

suspend fun sumOf(vararg inputs: NumberNDArrayCore) = inputs.sum()
