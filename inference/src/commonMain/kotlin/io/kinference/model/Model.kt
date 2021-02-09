package io.kinference.model

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.onnx.ModelProto

class Model(proto: ModelProto) {
    val graph = Graph(proto.graph!!)

    fun predict(input: Collection<ONNXData>): List<ONNXData> {
        return graph.execute(input.toList())
    }

    companion object {
        fun load(bytes: ByteArray): Model {
            val modelScheme = ModelProto.ADAPTER.decode(bytes)
            return Model(modelScheme)
        }
    }
}

expect fun Model.Companion.load(file: String): Model
