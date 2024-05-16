@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions.gather

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.computeGatherShape
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray

@MakePublic
@GenerateNameFromPrimitives
internal suspend fun gatherByBlocksPrimitive(array: PrimitiveNDArray, indices: NDArrayCore, axis: Int = 0): PrimitiveNDArray {
    val actualIndices = indices.toIntNDArray()
    val actualAxis = array.indexAxis(axis)

    val destShape = computeGatherShape(array.shape, actualAxis, indices)

    val dataBatchSize = array.computeBlockSize(toDim = actualAxis)
    val indicesSize = actualIndices.linearSize

    val dataToCopySize = array.computeBlockSize(fromDim = actualAxis + 1)
    val dataToCopyBlocks = dataToCopySize / array.array.blockSize

    val dataBlocks = array.array.blocks
    val dataMarkers = array.array.marker

    val destBatchBlocksOffset = indicesSize * dataToCopyBlocks
    val inputBatchBlockOffset = array.shape[actualAxis] * dataToCopyBlocks

    val destArray = arrayOfNulls<PrimitiveArray>(destBatchBlocksOffset * dataBatchSize)
    val destMarkersArray = arrayOfNulls<StateMarker>(destBatchBlocksOffset * dataBatchSize)


    for (dataBatchNum in 0 until dataBatchSize) {
        val dataBlocksOffset = inputBatchBlockOffset * dataBatchNum

        var destBlocksOffset = destBatchBlocksOffset * dataBatchNum

        val indicesPointer = actualIndices.array.pointer()
        indicesPointer.forEach(indicesSize) { idx ->
            val dataOffset = dataBlocksOffset + idx * dataToCopyBlocks

            for (blockIdx in 0 until dataToCopyBlocks) {
                destArray[destBlocksOffset + blockIdx] = dataBlocks[dataOffset + blockIdx]
                destMarkersArray[destBlocksOffset + blockIdx] = dataMarkers[dataOffset + blockIdx]
            }

            destBlocksOffset += dataToCopyBlocks
        }
    }

    return PrimitiveNDArray(PrimitiveTiledArray(destArray as Array<PrimitiveArray>, destMarkersArray as Array<StateMarker>), Strides(destShape))
}
