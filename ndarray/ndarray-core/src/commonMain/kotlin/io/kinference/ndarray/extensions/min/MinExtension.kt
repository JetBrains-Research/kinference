package io.kinference.ndarray.extensions.min

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun List<NumberNDArrayCore>.min(): NumberNDArrayCore {
    require(isNotEmpty()) { "Input array must have at least one element" }
    if (size == 1) return single()

    val inputType = this.first().type
    require(this.all { it.type == inputType }) { "Input tensors should have the same data type" }

    return when (inputType) {
        DataType.DOUBLE -> (this as List<DoubleNDArray>).min()
        DataType.FLOAT -> (this as List<FloatNDArray>).min()
        DataType.BYTE -> (this as List<ByteNDArray>).min()
        DataType.SHORT -> (this as List<ShortNDArray>).min()
        DataType.INT -> (this as List<IntNDArray>).min()
        DataType.LONG -> (this as List<LongNDArray>).min()
        DataType.UBYTE -> (this as List<UByteNDArray>).min()
        DataType.USHORT -> (this as List<UShortNDArray>).min()
        DataType.UINT -> (this as List<UIntNDArray>).min()
        DataType.ULONG -> (this as List<ULongNDArray>).min()
        else -> error("Min operation is only applicable to numeric tensors, current type: $inputType")
    }
}

suspend fun Array<out NumberNDArrayCore>.min() = this.toList().min()

suspend fun minOf(vararg inputs: NumberNDArrayCore) = inputs.min()
