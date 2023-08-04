package io.kinference.ndarray.extensions.reduce.logSum

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.reduceLogSum(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceLogSum(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceLogSum(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceLogSum(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceLogSum(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceLogSum(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceLogSum(axes, keepDims)
        else -> error("Unsupported type for reduceLogSum operation, supported only number types, current = $type")
    }
}
