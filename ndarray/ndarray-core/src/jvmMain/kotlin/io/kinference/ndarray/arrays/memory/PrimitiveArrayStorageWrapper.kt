@file:GeneratePrimitives(DataType.ALL)
package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.blockSizeByStrides
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray

@GenerateNameFromPrimitives
internal object PrimitiveArrayStorageWrapper {
    private val type = DataType.CurrentPrimitive

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
