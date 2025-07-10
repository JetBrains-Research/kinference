package io.kinference.core.model

import io.kinference.core.*
import io.kinference.core.graph.KIGraph
import io.kinference.graph.Contexts
import io.kinference.model.Model
import io.kinference.ndarray.arrays.memory.*
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.*
import kotlinx.atomicfu.atomic
import kotlinx.coroutines.*

class KIModel(
    val id: String,
    val name: String,
    val opSet: OperatorSetRegistry,
    val graph: KIGraph,
    predictionConfig: PredictionConfig = PredictionConfigs.NoAllocator,
) : Model<KIONNXData<*>>, Profilable, Cacheable {
    private val profiles: MutableList<ProfilingContext> = ArrayList()
    private val predictionContextDispatcher: PredictionContextDispatcher = PredictionContextDispatcher(predictionConfig)

    override fun addProfilingContext(name: String): ProfilingContext = ProfilingContext(name).apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override suspend fun predict(input: List<KIONNXData<*>>, profile: Boolean): Map<String, KIONNXData<*>> {
        val contexts = Contexts<KIONNXData<*>>(
            null,
            if (profile) addProfilingContext("Model $name") else null
        )

        val results = ResourcesDispatcher.reserveCore().suspendUse {
            val predictionContext = predictionContextDispatcher.getPredictionContext()
            val output = if (predictionContextDispatcher.allocationMode != AllocationMode.Auto) withContext(predictionContext) {
                return@withContext graph.execute(input, contexts)
            } else withContext(predictionContext) {
                return@withContext graph.execute(input, contexts).map { it.clone(it.name) }.toList()
            }

            predictionContextDispatcher.returnStorage(predictionContext)
            output
        }

        return results.associateBy { it.name!! }
    }

    override fun close() {
        graph.close()
        predictionContextDispatcher.close()
    }

    override fun clearCache() {
        predictionContextDispatcher.clearCache()
    }

    companion object {
        private val modelCounter = atomic(0)

        private fun generateModelId(): Int = modelCounter.incrementAndGet()

        suspend operator fun invoke(
            proto: ModelProto,
            predictionConfig: PredictionConfig = PredictionConfigs.NoAllocator,
        ): KIModel {
            val name = "${proto.domain}:${proto.modelVersion}"
            val id = "$name:${generateModelId()}"
            val opSet = OperatorSetRegistry(proto.opSetImport)
            val graph = KIGraph(proto.graph!!, opSet)
            return KIModel(id, name, opSet, graph, predictionConfig)
        }
    }
}
