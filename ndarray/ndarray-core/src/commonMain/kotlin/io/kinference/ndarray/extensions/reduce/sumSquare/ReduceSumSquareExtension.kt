package io.kinference.ndarray.extensions.reduce.sumSquare

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

fun NumberNDArrayCore.reduceSumSquare(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceSumSquare(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceSumSquare(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceSumSquare(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceSumSquare(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceSumSquare(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceSumSquare(axes, keepDims)
        else -> error("Unsupported type for reduceSumSquare operation, supported only number types, current = $type")
    }
}
