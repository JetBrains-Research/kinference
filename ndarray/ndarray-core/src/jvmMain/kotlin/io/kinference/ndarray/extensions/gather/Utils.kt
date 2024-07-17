package io.kinference.ndarray.extensions.gather

import io.kinference.ndarray.arrays.*

internal suspend fun NDArrayCore.toIntNDArray(): IntNDArray {
    return when (this) {
        is IntNDArray -> this
        is LongNDArray -> {
            val dest = IntNDArray(this.shape)
            for (blockIdx in this.array.blocks.indices) {
                val srcBlock = this.array.blocks[blockIdx]
                val destBlock = dest.array.blocks[blockIdx]

                for (idx in destBlock.indices) {
                    destBlock[idx] = srcBlock[idx].toInt()
                }
            }

            return dest
        }

        else -> throw UnsupportedOperationException()
    }
}
