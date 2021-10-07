package io.kinference.core.model

import io.kinference.core.data.KIONNXData
import io.kinference.data.ONNXData
import io.kinference.core.graph.*
import io.kinference.model.Model
import io.kinference.protobuf.message.ModelProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class KIModel(proto: ModelProto) : Model {
    val graph = Graph(proto.graph!!)
    val name: String = "${proto.domain}:${proto.modelVersion}"

    private val profiles: MutableList<ProfilingContext> = ArrayList()
    fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    fun resetProfiles() = profiles.clear()

    override fun predict(input: List<ONNXData<*>>, profile: Boolean): List<ONNXData<*>> {
        val context = if (profile) ProfilingContext("Model $name").apply { profiles.add(this) } else null
        return graph.execute(input as List<KIONNXData<*>>, profilingContext = context)
    }
}
