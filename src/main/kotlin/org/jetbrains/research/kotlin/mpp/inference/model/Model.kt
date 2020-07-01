package org.jetbrains.research.kotlin.mpp.inference.model

import ModelProto
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.graph.Graph
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
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
