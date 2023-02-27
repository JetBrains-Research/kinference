package io.kinference.graph

import io.kinference.data.ONNXData
import io.kinference.profiler.ProfilingContext

class Contexts<T : ONNXData<*, *>>(
    val graph: GraphContext<T>?, 
    val profiling: ProfilingContext?
)

fun <T : ONNXData<*, *>> emptyContexts() = Contexts<T>(null, null)
