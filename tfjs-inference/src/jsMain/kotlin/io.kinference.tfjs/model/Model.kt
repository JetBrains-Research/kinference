package io.kinference.tfjs.model

import io.kinference.tfjs.data.ONNXData
import io.kinference.tfjs.graph.Graph
import io.kinference.protobuf.message.ModelProto
import io.kinference.tfjs.utils.setDefaultBackend
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
