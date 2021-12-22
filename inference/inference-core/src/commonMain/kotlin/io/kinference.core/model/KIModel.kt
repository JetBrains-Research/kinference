package io.kinference.core.model

import io.kinference.core.KIONNXData
import io.kinference.core.graph.*
import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class KIModel(proto: ModelProto) : Model<KIONNXData<*>>, Profilable {
    val name: String = "${proto.domain}:${proto.modelVersion}"
    private val opSet = OperatorSetRegistry(proto.opSetImport)
    val graph = KIGraph(proto.graph!!, opSet)

    private val profiles: MutableList<ProfilingContext> = ArrayList()
    override fun addContext(name: String): ProfilingContext = ProfilingContext(name).apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override fun predict(input: List<KIONNXData<*>>, profile: Boolean, checkCancelled: () -> Unit): Map<String, KIONNXData<*>> {
        val context = if (profile) addContext("Model $name") else null
        val execResult = graph.execute(input, profilingContext = context, checkCancelled = checkCancelled)
        return execResult.associateBy { it.name!! }
    }
}
