package org.jetbrains.research.kotlin.inference.model

import ModelProto
import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.graph.Graph
import java.io.File

class Model(proto: ModelProto) {
    val graph = Graph(proto.graph!!)

    inline fun <reified T : Number> predict(input: List<T>): List<ONNXData> {
        return graph.setInput(input).execute()
    }

    fun predict(input: Collection<ONNXData>): List<ONNXData> {
        input.forEach { graph.setInput(it) }
        return graph.execute()
    }

    companion object {
        fun load(file: String): Model {
            val modelScheme = ModelProto.ADAPTER.decode(File(file).readBytes())
            return Model(modelScheme)
        }
    }
}
