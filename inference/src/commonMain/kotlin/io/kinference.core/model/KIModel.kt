package io.kinference.core.model

import io.kinference.core.graph.*
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataAdapter
import io.kinference.model.Model
import io.kinference.protobuf.message.ModelProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class KIModel<T>(proto: ModelProto, private val adapter: ONNXDataAdapter<T, ONNXData<*>>) : Model<T> {
    val graph = Graph(proto.graph!!)
    val name: String = "${proto.domain}:${proto.modelVersion}"

    private val profiles: MutableList<ProfilingContext> = ArrayList()
    fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    fun resetProfiles() = profiles.clear()

    override fun predict(input: List<T>, profile: Boolean): List<T> {
        val context = if (profile) ProfilingContext("Model $name").apply { profiles.add(this) } else null
        val onnxInput = input.map { adapter.toONNXData(it) }
        val execResult = graph.execute(onnxInput, profilingContext = context)
        return execResult.map { adapter.fromONNXData(it) }
    }
}
