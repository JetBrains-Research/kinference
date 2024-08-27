package io.kinference.ndarray.arrays.memory.contexts

import io.kinference.ndarray.arrays.memory.ArrayStorage
import kotlin.coroutines.CoroutineContext

interface BaseAllocatorContext: CoroutineContext.Element

abstract class BaseAllocatorContextWithStorage<T : ArrayStorage>(protected val storage: T) : BaseAllocatorContext {
    fun finalizeContext() {
        storage.resetState()
    }
}

fun CoroutineContext.finalizeAllocatorContext() {
    this.fold(Unit) { _, context ->
        if (context is BaseAllocatorContextWithStorage<*>)
            context.finalizeContext()
    }
}

class NoAllocatorContext : BaseAllocatorContext {
    companion object Key : CoroutineContext.Key<NoAllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key
}
