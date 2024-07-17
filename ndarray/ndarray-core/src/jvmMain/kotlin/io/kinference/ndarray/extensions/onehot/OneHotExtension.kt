package io.kinference.ndarray.extensions.onehot

import io.kinference.ndarray.arrays.*

internal suspend fun NumberNDArrayCore.getOneHotIndices(depth: Int) = when (this) {
    is IntNDArray -> getOneHotIndices(this, depth)
    is FloatNDArray -> getOneHotIndices(this, depth)
    is DoubleNDArray -> getOneHotIndices(this, depth)
    is LongNDArray -> getOneHotIndices(this, depth)
    is ShortNDArray -> getOneHotIndices(this, depth)
    is ByteNDArray -> getOneHotIndices(this, depth)
    is UIntNDArray -> getOneHotIndices(this, depth)
    is UShortNDArray -> getOneHotIndices(this, depth)
    is UByteNDArray -> getOneHotIndices(this, depth)
    is ULongNDArray -> getOneHotIndices(this, depth)
    else -> error("OneHot indices array must have numeric data type, current type: $type")
}

suspend fun oneHot(indices: NumberNDArrayCore, depth: Int, values: NDArrayCore, axis: Int = -1): NDArrayCore {
    return when (values) {
        is FloatNDArray -> FloatNDArray.oneHot(axis, indices, depth, values)
        is DoubleNDArray -> DoubleNDArray.oneHot(axis, indices, depth, values)
        is IntNDArray -> IntNDArray.oneHot(axis, indices, depth, values)
        is LongNDArray -> LongNDArray.oneHot(axis, indices, depth, values)
        is ShortNDArray -> ShortNDArray.oneHot(axis, indices, depth, values)
        is ByteNDArray -> ByteNDArray.oneHot(axis, indices, depth, values)
        is UIntNDArray -> UIntNDArray.oneHot(axis, indices, depth, values)
        is ULongNDArray -> ULongNDArray.oneHot(axis, indices, depth, values)
        is UShortNDArray -> UShortNDArray.oneHot(axis, indices, depth, values)
        is UByteNDArray -> UByteNDArray.oneHot(axis, indices, depth, values)
        is BooleanNDArray -> BooleanNDArray.oneHot(axis, indices, depth, values)
        else -> error("Unsupported \"values\" data type: ${values.type}")
    }
}
