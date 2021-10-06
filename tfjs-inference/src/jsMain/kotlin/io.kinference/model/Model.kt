package io.kinference.model

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.setDefaultBackend
import kotlin.time.ExperimentalTime

@ExperimentalTime
class Model(proto: ModelProto) {
    init {
        setDefaultBackend()
    }

    val graph = Graph(proto.graph!!)
    val name: String = "${proto.domain}:${proto.modelVersion}"

    fun predict(input: Collection<ONNXData>): List<ONNXData> {
        return graph.execute(input.toList())
    }

    companion object {
        fun load(bytes: ByteArray): Model {
            val modelScheme = ModelProto.decode(bytes)
            return Model(modelScheme)
        }
    }
}
