@file:GeneratePrimitives(DataType.ALL)
package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.blockSizeByStrides
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import io.kinference.utils.inlines.InlineInt

@GenerateNameFromPrimitives
internal class PrimitiveArrayStorageWrapper {
    private val type = DataType.CurrentPrimitive

    private val storage = HashMap<InlineInt, ArrayDeque<PrimitiveArray>>(2)

    fun getNDArray(strides: Strides, fillZeros: Boolean = false): MutablePrimitiveNDArray {
        val blockSize = InlineInt(blockSizeByStrides(strides))
        val blocksNum = strides.linearSize / blockSize.value

        val queue = storage.getOrPut(blockSize) { ArrayDeque(blocksNum) }

        val blocks = Array(blocksNum) {
            val block = queue.removeFirstOrNull()
            if (fillZeros) {
                block?.fill(PrimitiveConstants.ZERO)
            }
            block ?: PrimitiveArray(blockSize.value)
        }

        val tiled = PrimitiveTiledArray(blocks)

        return MutablePrimitiveNDArray(tiled, strides)
    }

    fun returnNDArray(ndarray: PrimitiveNDArray) {
        val blockSize = InlineInt(ndarray.array.blockSize)
        val blocksNum = ndarray.array.blocksNum

        val queue = storage.getOrPut(blockSize) { ArrayDeque(blocksNum) }

        queue.addAll(ndarray.array.blocks)
    }

    fun getNDArray(strides: Strides, storage: SingleArrayStorage, fillZeros: Boolean = false): MutablePrimitiveNDArray {
        val blockSize = blockSizeByStrides(strides)
        val blocksNum = strides.linearSize / blockSize
        val blocks = Array(blocksNum) { storage.getArray(type, blockSize, fillZeros) }
        val typedBlocks = blocks.map { it as PrimitiveArray }.toTypedArray()
        val tiled = PrimitiveTiledArray(typedBlocks)

        return MutablePrimitiveNDArray(tiled, strides)
    }

    fun returnNDArray(storage: SingleArrayStorage, ndarray: PrimitiveNDArray) {
        val blockSize = ndarray.array.blockSize
        storage.returnArrays(type, blockSize, ndarray.array.blocks as Array<Any>)
    }
}
