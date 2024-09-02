package io.kinference.ndarray.arrays.memory.contexts

import io.kinference.ndarray.arrays.memory.storage.AutoArrayHandlingStorage
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import io.kinference.utils.*
import kotlinx.coroutines.CoroutineDispatcher
import kotlin.coroutines.*

internal class AutoAllocatorContext internal constructor(
    dispatcher: CoroutineDispatcher,
    storage: AutoArrayHandlingStorage,
) : AllocatorContext<AutoArrayHandlingStorage>(dispatcher, storage)
