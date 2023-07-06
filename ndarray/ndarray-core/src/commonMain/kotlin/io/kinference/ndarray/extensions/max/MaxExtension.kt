package io.kinference.ndarray.extensions.max

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun List<NumberNDArrayCore>.max(): NumberNDArrayCore {
    if (isEmpty()) error("Array for max operation must have at least one element")
    if (size == 1) return single()

    val inputType = this.first().type
    require(this.all { it.type == inputType }) { "Input tensors must have the same data type" }

    return when (inputType) {
        DataType.DOUBLE -> (this as List<DoubleNDArray>).max()
        DataType.FLOAT -> (this as List<FloatNDArray>).max()
        DataType.BYTE -> (this as List<ByteNDArray>).max()
        DataType.SHORT -> (this as List<ShortNDArray>).max()
        DataType.INT -> (this as List<IntNDArray>).max()
        DataType.LONG -> (this as List<LongNDArray>).max()
        DataType.UBYTE -> (this as List<UByteNDArray>).max()
        DataType.USHORT -> (this as List<UShortNDArray>).max()
        DataType.UINT -> (this as List<UIntNDArray>).max()
        DataType.ULONG -> (this as List<ULongNDArray>).max()
        else -> error("Unsupported data type in max operation, tensors must have number data type, current is $inputType")
    }
}

suspend fun Array<out NumberNDArrayCore>.max() = this.toList().max()

suspend fun maxOf(vararg inputs: NumberNDArrayCore) = inputs.max()
