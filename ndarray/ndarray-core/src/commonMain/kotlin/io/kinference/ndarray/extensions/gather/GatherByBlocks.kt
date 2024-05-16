package io.kinference.ndarray.extensions.gather

import io.kinference.ndarray.arrays.*

internal suspend fun NDArrayCore.gatherByBlocks(indices: NDArrayCore, axis: Int = 0): NDArrayCore {
    return when(this) {
        is FloatNDArray -> gatherByBlocksFloat(this, indices, axis)
        is BooleanNDArray -> gatherByBlocksBoolean(this, indices, axis)
        is ByteNDArray -> gatherByBlocksByte(this, indices, axis)
        is DoubleNDArray -> gatherByBlocksDouble(this, indices, axis)
        is IntNDArray -> gatherByBlocksInt(this, indices, axis)
        is LongNDArray -> gatherByBlocksLong(this, indices, axis)
        is ShortNDArray -> gatherByBlocksShort(this, indices, axis)
        is UByteNDArray -> gatherByBlocksUByte(this, indices, axis)
        is UIntNDArray -> gatherByBlocksUInt(this, indices, axis)
        is ULongNDArray -> gatherByBlocksULong(this, indices, axis)
        is UShortNDArray -> gatherByBlocksUShort(this, indices, axis)
        else -> throw UnsupportedOperationException()
    }
}
