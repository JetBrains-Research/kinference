package io.kinference.ndarray.extensions.reduce.max

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

fun NumberNDArrayCore.reduceMax(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceMax(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceMax(axes, keepDims)
        DataType.BYTE -> (this as ByteNDArray).reduceMax(axes, keepDims)
        DataType.SHORT -> (this as ShortNDArray).reduceMax(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceMax(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceMax(axes, keepDims)
        DataType.UBYTE -> (this as UByteNDArray).reduceMax(axes, keepDims)
        DataType.USHORT -> (this as UShortNDArray).reduceMax(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceMax(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceMax(axes, keepDims)
        else -> error("Unsupported type for reduceMax operation, supported only number types, current = $type")
    }
}
