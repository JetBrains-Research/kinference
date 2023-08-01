package io.kinference.ndarray.extensions.reduce.min

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

fun NumberNDArrayCore.reduceMin(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceMin(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceMin(axes, keepDims)
        DataType.BYTE -> (this as ByteNDArray).reduceMin(axes, keepDims)
        DataType.SHORT -> (this as ShortNDArray).reduceMin(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceMin(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceMin(axes, keepDims)
        DataType.UBYTE -> (this as UByteNDArray).reduceMin(axes, keepDims)
        DataType.USHORT -> (this as UShortNDArray).reduceMin(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceMin(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceMin(axes, keepDims)
        else -> error("Unsupported type for reduceMax operation, supported only number types, current = $type")
    }
}
