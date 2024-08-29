@file:GeneratePrimitives(DataType.ALL)
package io.kinference.ndarray.arrays.memory.storage

import io.kinference.ndarray.INIT_STORAGE_SIZE
import io.kinference.ndarray.arrays.memory.MemoryManager
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.extensions.utils.getOrPut
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap

@GenerateNameFromPrimitives
internal class PrimitiveAutoHandlingArrayStorage : TypedAutoHandlingStorage {
    private val used = Int2ObjectOpenHashMap<ArrayDeque<PrimitiveArray>>(INIT_STORAGE_SIZE)
    private val unused = Int2ObjectOpenHashMap<ArrayDeque<PrimitiveArray>>(INIT_STORAGE_SIZE)

    companion object {
        private val type = DataType.CurrentPrimitive
    }

    internal fun getBlock(blocksNum: Int, blockSize: Int, limiter: MemoryManager): Array<PrimitiveArray> {
        val unusedQueue = unused.getOrPut(blockSize) { ArrayDeque(blocksNum) }
        val usedQueue = used.getOrPut(blockSize) { ArrayDeque(blocksNum) }

        val blocks = if (limiter.checkMemoryLimitAndAdd(type, size = blockSize * blocksNum)) {
            Array(blocksNum) {
                unusedQueue.removeFirstOrNull()?.apply {
                    fill(PrimitiveConstants.ZERO)
                } ?: PrimitiveArray(blockSize)
            }
        } else {
            Array(blocksNum) { PrimitiveArray(blockSize) }
        }

        usedQueue.addAll(blocks)

        return blocks
    }

    override fun moveBlocksIntoUnused() {
        used.forEach { (blockSize, queue) ->
            unused[blockSize]!!.addAll(queue)
            queue.clear()
        }
    }
}
