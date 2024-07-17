package io.kinference.ndarray.extensions.reduce.prod

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.reduceProd(axes: IntArray, keepDims: Boolean): NumberNDArrayCore {
    return when(type) {
        DataType.FLOAT -> (this as FloatNDArray).reduceProd(axes, keepDims)
        DataType.DOUBLE -> (this as DoubleNDArray).reduceProd(axes, keepDims)
        DataType.INT -> (this as IntNDArray).reduceProd(axes, keepDims)
        DataType.LONG -> (this as LongNDArray).reduceProd(axes, keepDims)
        DataType.UINT -> (this as UIntNDArray).reduceProd(axes, keepDims)
        DataType.ULONG -> (this as ULongNDArray).reduceProd(axes, keepDims)
        else -> error("Unsupported type for reduceProd operation, supported only number types, current = $type")
    }
}
