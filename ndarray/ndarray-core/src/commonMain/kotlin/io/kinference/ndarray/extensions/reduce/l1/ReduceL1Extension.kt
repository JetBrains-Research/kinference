package io.kinference.ndarray.extensions.reduce.l1

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.reduceL1(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceL1(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceL1(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceL1(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceL1(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceL1(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceL1(axes, keepDims)
        else -> error("Unsupported type for reduceL1 operation, supported only number types, current = $type")
    }
}
