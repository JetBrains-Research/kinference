package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.coroutines.CoroutineContext

data class AutoAllocatorContext internal constructor(
    private val storage: ArrayStorage,
    private val returnStorageFn: (ArrayStorage) -> Unit
) : CoroutineContext.Element {

    companion object Key : CoroutineContext.Key<AutoAllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key

    internal fun getArrays(type: DataType, size: Int, count: Int): Array<Any> {
        return Array(count) { storage.getArray(type, size) }
    }

    fun returnUsedArrays() {
        storage.moveArrays()
        returnStorageFn(storage)
    }
}
