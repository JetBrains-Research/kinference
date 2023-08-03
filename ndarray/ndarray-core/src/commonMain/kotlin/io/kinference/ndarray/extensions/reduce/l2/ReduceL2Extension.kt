package io.kinference.ndarray.extensions.reduce.l2

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.reduceL2(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceL2(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceL2(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceL2(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceL2(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceL2(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceL2(axes, keepDims)
        else -> error("Unsupported type for reduceL2 operation, supported only number types, current = $type")
    }
}
