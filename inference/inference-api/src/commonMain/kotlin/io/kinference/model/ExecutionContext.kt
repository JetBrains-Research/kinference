package io.kinference.model

import kotlin.coroutines.CoroutineContext

/**
 * Execution context for KInference models passed to [Model.predict].
 * @property coroutineContext coroutine context used for model computations.
 * @property checkCancelled check if the computation was cancelled, and, if yes, interrupts model execution.
 */
class ExecutionContext(
    val coroutineContext: CoroutineContext,
    val checkCancelled: () -> Unit = { }
)
