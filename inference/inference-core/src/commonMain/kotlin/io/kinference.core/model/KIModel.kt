package io.kinference.core.model

import io.kinference.core.KIONNXData
import io.kinference.core.graph.KIGraph
import io.kinference.graph.Contexts
import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto

class KIModel(val name: String, val opSet: OperatorSetRegistry, val graph: KIGraph) : Model<KIONNXData<*>>, Profilable {
    private val profiles: MutableList<ProfilingContext> = ArrayList()
    override fun addProfilingContext(name: String): ProfilingContext = ProfilingContext(name).apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override suspend fun predict(input: List<KIONNXData<*>>, profile: Boolean): Map<String, KIONNXData<*>> {
        val contexts = Contexts<KIONNXData<*>>(
            null,
            if (profile) addProfilingContext("Model $name") else null
        )
        val execResult = graph.execute(input, contexts)
        return execResult.associateBy { it.name!! }
    }

    override fun close() {
        graph.close()
    }

    companion object {
        suspend operator fun invoke(proto: ModelProto): KIModel {
            val name = "${proto.domain}:${proto.modelVersion}"
            val opSet = OperatorSetRegistry(proto.opSetImport)
            val graph = KIGraph(proto.graph!!, opSet)
            return KIModel(name, opSet, graph)
        }
    }
}
