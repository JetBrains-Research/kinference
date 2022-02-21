package io.kinference.core.model

import io.kinference.core.KIEngine
import io.kinference.core.KIONNXData
import io.kinference.core.graph.*
import io.kinference.graph.Contexts
import io.kinference.model.ExecutionContext
import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.optimizer.GraphOptimizer
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class KIModel(proto: ModelProto, optimize: Boolean) : Model<KIONNXData<*>>, Profilable {
    val name: String = "${proto.domain}:${proto.modelVersion}"
    private val opSet = OperatorSetRegistry(proto.opSetImport)
    val graph: KIGraph

    init {
        var graph = KIGraph(proto.graph!!, opSet)
        if (optimize) {
            graph = GraphOptimizer(graph).run(KIEngine.optimizerRules) as KIGraph
        }
        this.graph = graph
    }

    private val profiles: MutableList<ProfilingContext> = ArrayList()
    override fun addProfilingContext(name: String): ProfilingContext = ProfilingContext(name).apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override fun predict(input: List<KIONNXData<*>>, profile: Boolean, executionContext: ExecutionContext?): Map<String, KIONNXData<*>> {
        val contexts = Contexts<KIONNXData<*>>(
            null,
            if (profile) addProfilingContext("Model $name") else null,
            executionContext
        )
        val execResult = graph.execute(input, contexts)
        return execResult.associateBy { it.name!! }
    }
}
