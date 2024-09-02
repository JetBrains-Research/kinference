@file:GeneratePrimitives(DataType.ALL)
package io.kinference.ndarray.arrays.memory.storage

import io.kinference.ndarray.INIT_STORAGE_SIZE
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.memory.MemoryManager
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.blockSizeByStrides
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.extensions.utils.getOrPut
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap

@GenerateNameFromPrimitives
internal class PrimitiveManualHandlingArrayStorage : TypedManualHandlingStorage {
    private val storage = Int2ObjectOpenHashMap<ArrayDeque<PrimitiveArray>>(INIT_STORAGE_SIZE)

    companion object {
        private val type = DataType.CurrentPrimitive
    }

    override fun getNDArray(strides: Strides, fillZeros: Boolean, limiter: MemoryManager): MutableNDArrayCore {
        val blockSize = blockSizeByStrides(strides)
        val blocksNum = strides.linearSize / blockSize
        val blocks = if (limiter.checkMemoryLimitAndAdd(type.getPrimitiveArraySizeInBytes(arraySize = blockSize * blocksNum))) {
            val queue = storage.getOrPut(blockSize) { ArrayDeque(blocksNum) }
            Array(blocksNum) {
                queue.removeFirstOrNull()?.apply {
                    fill(PrimitiveConstants.ZERO)
                } ?: PrimitiveArray(blockSize)
            }
        } else {
            Array(blocksNum) { PrimitiveArray(blockSize) }
        }

        val tiled = PrimitiveTiledArray(blocks)

        return MutablePrimitiveNDArray(tiled, strides)
    }

    override fun returnNDArray(ndarray: NDArrayCore) {
        require(ndarray is PrimitiveNDArray)
        val blockSize = ndarray.array.blockSize
        val blocksNum = ndarray.array.blocksNum

        val queue = storage.getOrPut(blockSize) { ArrayDeque(blocksNum) }

        queue.addAll(ndarray.array.blocks)
    }

    override fun clear() {
        storage.forEach { (_, queue) -> queue.clear() }
    }
}
