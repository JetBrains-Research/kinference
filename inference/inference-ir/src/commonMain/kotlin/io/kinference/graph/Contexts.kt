package io.kinference.graph

import io.kinference.data.ONNXData
import io.kinference.model.ExecutionContext
import io.kinference.profiler.ProfilingContext
import kotlin.coroutines.EmptyCoroutineContext

class Contexts<T : ONNXData<*, *>>(
    val graph: GraphContext<T>?, 
    val profiling: ProfilingContext?,
    val execution: ExecutionContext?
)

fun <T : ONNXData<*, *>> emptyContexts() = Contexts<T>(null, null, null)
fun ExecutionContext?.asCoroutineContext() = this?.coroutineContext ?: EmptyCoroutineContext
