package io.kinference.ndarray.arrays.memory.contexts

import io.kinference.ndarray.arrays.memory.storage.AutoArrayHandlingStorage
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import io.kinference.utils.*
import kotlinx.coroutines.CoroutineDispatcher
import kotlin.coroutines.*

@OptIn(ExperimentalStdlibApi::class)
internal class AutoAllocatorContext internal constructor(
    dispatcher: CoroutineDispatcher,
    storage: AutoArrayHandlingStorage,
) : AllocatorContext<AutoArrayHandlingStorage>(dispatcher, storage) {
    companion object Key : AbstractCoroutineContextKey<AllocatorContext<*>, AutoAllocatorContext>(
        AllocatorContext.Key, { it as? AutoAllocatorContext }
    )
}
