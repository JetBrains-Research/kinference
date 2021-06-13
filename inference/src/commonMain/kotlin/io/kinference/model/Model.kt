package io.kinference.model

import io.kinference.data.ONNXData
import io.kinference.graph.*
import io.kinference.protobuf.message.ModelProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class Model(proto: ModelProto) {
    val graph = Graph(proto.graph!!)
    val name: String = "${proto.domain}:${proto.modelVersion}"

    private val profiles: MutableList<ProfilingContext> = ArrayList()
    fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    fun resetProfiles() = profiles.clear()

    fun predict(input: Collection<ONNXData>, profile: Boolean = false): List<ONNXData> {
        val context = if (profile) ProfilingContext("Model $name").apply { profiles.add(this) } else null
        return graph.execute(input.toList(), profilingContext = context)
    }

    companion object {
        fun load(bytes: ByteArray): Model {
            val modelScheme = ModelProto.decode(bytes)
            return Model(modelScheme)
        }
    }
}
