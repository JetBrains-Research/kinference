package io.kinference.core.model

import io.kinference.core.KIONNXData
import io.kinference.core.graph.KIGraph
import io.kinference.core.markOutput
import io.kinference.graph.Contexts
import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto
import io.kinference.ndarray.arrays.memory.AllocatorContext
import kotlinx.coroutines.withContext
import kotlinx.atomicfu.atomic

class KIModel(val id: String, val name: String, val opSet: OperatorSetRegistry, val graph: KIGraph, val useAllocator: Boolean = true) : Model<KIONNXData<*>>, Profilable {
    private val inferenceCycleCounter = atomic(0L)
    private val profiles: MutableList<ProfilingContext> = ArrayList()
    override fun addProfilingContext(name: String): ProfilingContext = ProfilingContext(name).apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override suspend fun predict(input: List<KIONNXData<*>>, profile: Boolean): Map<String, KIONNXData<*>> {
        val contexts = Contexts<KIONNXData<*>>(
            null,
            if (profile) addProfilingContext("Model $name") else null
        )

        val results = if (useAllocator) {
            withContext(AllocatorContext(id, getInferenceCycleId())) {
                val coroutineContext = coroutineContext[AllocatorContext.Key]!!
                val execResult = graph.execute(input, contexts)
                execResult.forEach { it.markOutput() }
                coroutineContext.closeAllocated()
                execResult
            }
        } else {
            graph.execute(input, contexts)
        }

        return results.associateBy { it.name!! }
    }


    override suspend fun close() {
        graph.close()
    }

    private fun getInferenceCycleId(): Long = inferenceCycleCounter.incrementAndGet()

    companion object {
        private val modelCounter = atomic(0)

        private fun generateModelId(): Int = modelCounter.incrementAndGet()

        suspend operator fun invoke(proto: ModelProto, useAllocator: Boolean = true): KIModel {
            val name = "${proto.domain}:${proto.modelVersion}"
            val id = "$name:${generateModelId()}"
            val opSet = OperatorSetRegistry(proto.opSetImport)
            val graph = KIGraph(proto.graph!!, opSet)
            return KIModel(id, name, opSet, graph, useAllocator)
        }
    }
}
