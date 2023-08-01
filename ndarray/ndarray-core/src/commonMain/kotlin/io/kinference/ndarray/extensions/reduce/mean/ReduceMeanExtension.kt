package io.kinference.ndarray.extensions.reduce.mean

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.reduceMean(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceMean(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceMean(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceMean(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceMean(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceMean(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceMean(axes, keepDims)
        else -> error("Unsupported type for reduceMean operation, supported only 32/64-bit types, current = $type")
    }
}
