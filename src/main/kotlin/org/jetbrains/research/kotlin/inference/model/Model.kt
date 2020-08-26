package org.jetbrains.research.kotlin.inference.model

import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.graph.Graph
import org.jetbrains.research.kotlin.inference.onnx.ModelProto
import java.io.File

class Model(proto: ModelProto) {
    val graph = Graph(proto.graph!!)

    fun predict(input: Collection<ONNXData>): List<ONNXData> {
        return graph.execute(input.toList())
    }

    companion object {
        fun load(file: String): Model = load(File(file).readBytes())

        fun load(bytes: ByteArray): Model {
            val modelScheme = ModelProto.ADAPTER.decode(bytes)
            return Model(modelScheme)
        }
    }
}
