package io.kinference.ndarray.arrays.memory.contexts

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.storage.ManualArrayHandlingStorage
import io.kinference.ndarray.arrays.memory.storage.ManualStorage
import io.kinference.primitives.types.DataType
import io.kinference.utils.AllocatorContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlin.coroutines.AbstractCoroutineContextKey

@OptIn(ExperimentalStdlibApi::class)
class ManualAllocatorContext internal constructor(
    dispatcher: CoroutineDispatcher,
    storage: ManualArrayHandlingStorage,
) : AllocatorContext<ManualStorage>(dispatcher, storage) {
    companion object Key : AbstractCoroutineContextKey<AllocatorContext<*>, ManualAllocatorContext>(
        AllocatorContext.Key, { it as? ManualAllocatorContext }
    )


    fun getNDArray(dataType: DataType, strides: Strides, fillZeros: Boolean = false): MutableNDArrayCore {
        return storage.getNDArray(dataType, strides, fillZeros)
    }

    fun returnNDArray(ndArray: NDArrayCore) {
        storage.returnNDArray(ndArray)
    }
}
