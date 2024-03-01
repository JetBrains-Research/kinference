package io.kinference.utils

import kotlin.coroutines.CoroutineContext

data class ModelContext(val modelName: String) : CoroutineContext.Element {
    companion object Key : CoroutineContext.Key<ModelContext>
    override val key: CoroutineContext.Key<*> get() = Key
}
