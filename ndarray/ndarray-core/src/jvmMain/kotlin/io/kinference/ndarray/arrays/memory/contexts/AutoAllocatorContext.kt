package io.kinference.ndarray.arrays.memory.contexts

import io.kinference.ndarray.arrays.memory.storage.AutoArrayHandlingStorage
import io.kinference.primitives.types.DataType
import kotlin.coroutines.*

internal class AutoAllocatorContext internal constructor(
    storage: AutoArrayHandlingStorage,
) : BaseAllocatorContextWithStorage<AutoArrayHandlingStorage>(storage) {

    companion object Key : CoroutineContext.Key<AutoAllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key

    internal fun getArrays(type: DataType, size: Int, count: Int): Array<Any> {
        return storage.getArrays(type, size, count)
    }
}
