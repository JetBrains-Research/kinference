package io.kinference.ndarray.extensions.mean

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun List<NumberNDArrayCore>.mean(): NumberNDArrayCore {
    if (isEmpty()) error("Array for mean operation must have at least one element")
    if (size == 1) return single()

    val inputType = this.first().type
    require(this.all { it.type == inputType }) { "Input tensors must have the same data type" }

    return when (inputType) {
        DataType.DOUBLE -> (this as List<DoubleNDArray>).mean()
        DataType.FLOAT -> (this as List<FloatNDArray>).mean()
        else -> error("Unsupported data type in mean operation, tensors must have Float or Double data type, current is $inputType")
    }
}

suspend fun Array<out NumberNDArrayCore>.mean() = this.toList().mean()

suspend fun meanOf(vararg inputs: NumberNDArrayCore) = inputs.mean()
