@file:GeneratePrimitives(DataType.ALL)
@file:Suppress("DuplicatedCode")
package io.kinference.ndarray.arrays.memory.storage

import io.kinference.ndarray.arrays.memory.contexts.AutoAllocatorContext
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*

@GenerateNameFromPrimitives
internal fun AutoArrayHandlingStorage.getPrimitiveBlock(blocksNum: Int, blockSize: Int): Array<PrimitiveArray> {
    return (storage[DataType.CurrentPrimitive.ordinal] as PrimitiveAutoHandlingArrayStorage).getBlock(blocksNum = blocksNum, blockSize = blockSize, limiter = limiter)
}

@GenerateNameFromPrimitives
internal fun AutoAllocatorContext.getPrimitiveBlock(blocksNum: Int, blockSize: Int): Array<PrimitiveArray> {
    return storage.getPrimitiveBlock(blocksNum = blocksNum, blockSize = blockSize)
}
