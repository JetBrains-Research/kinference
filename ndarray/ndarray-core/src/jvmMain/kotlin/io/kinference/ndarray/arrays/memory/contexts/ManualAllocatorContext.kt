package io.kinference.ndarray.arrays.memory.contexts

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.storage.ManualArrayHandlingStorage
import io.kinference.ndarray.arrays.memory.storage.ManualStorage
import io.kinference.primitives.types.DataType
import kotlin.coroutines.CoroutineContext

class ManualAllocatorContext internal constructor(
    storage: ManualArrayHandlingStorage,
) : BaseAllocatorContextWithStorage<ManualStorage>(storage) {

    companion object Key : CoroutineContext.Key<ManualAllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key

    fun getNDArray(dataType: DataType, strides: Strides, fillZeros: Boolean = false): MutableNDArrayCore {
        return storage.getNDArray(dataType, strides, fillZeros)
    }

    fun returnNDArray(ndArray: NDArrayCore) {
        storage.returnNDArray(ndArray)
    }
}
