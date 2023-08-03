package io.kinference.ndarray.extensions.reduce.logSumExp

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.reduceLogSumExp(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceLogSumExp(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceLogSumExp(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceLogSumExp(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceLogSumExp(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceLogSumExp(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceLogSumExp(axes, keepDims)
        else -> error("Unsupported type for reduceLogSumExp operation, supported only number types, current = $type")
    }
}
