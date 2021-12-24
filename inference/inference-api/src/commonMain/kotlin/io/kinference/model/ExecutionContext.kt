package io.kinference.model

import kotlin.coroutines.CoroutineContext

class ExecutionContext(
    val coroutineContext: CoroutineContext,
    val checkCancelled: () -> Unit = { }
)
