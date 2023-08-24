package io.kinference.ndarray.extensions.onehot

import io.kinference.ndarray.arrays.*

internal fun NumberNDArrayCore.getOneHotIndices(depth: Int) = when (this) {
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
        is FloatNDArray -> oneHotFloat(axis, indices, depth, values)
        is DoubleNDArray -> oneHotDouble(axis, indices, depth, values)
        is IntNDArray -> oneHotInt(axis, indices, depth, values)
        is LongNDArray -> oneHotLong(axis, indices, depth, values)
        is ShortNDArray -> oneHotShort(axis, indices, depth, values)
        is ByteNDArray -> oneHotByte(axis, indices, depth, values)
        is UIntNDArray -> oneHotUInt(axis, indices, depth, values)
        is ULongNDArray -> oneHotULong(axis, indices, depth, values)
        is UShortNDArray -> oneHotUShort(axis, indices, depth, values)
        is UByteNDArray -> oneHotUByte(axis, indices, depth, values)
        is BooleanNDArray -> oneHotBoolean(axis, indices, depth, values)
        else -> error("Unsupported \"values\" data type: ${values.type}")
    }
}
