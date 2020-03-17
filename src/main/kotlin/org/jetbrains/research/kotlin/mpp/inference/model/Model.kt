package org.jetbrains.research.kotlin.mpp.inference.model

import ModelProto
import org.jetbrains.research.kotlin.mpp.inference.graph.Graph
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import java.io.File

class Model(proto: ModelProto) {
    val graph = Graph.build(proto.graph!!)

    inline fun <reified T: Number> predict(input: List<T>): List<Tensor<*>?> {
        return graph.setInput(input).run().fetchOutputs()
    }

    companion object {
        fun load(file: String): Model {
            val modelScheme = ModelProto.ADAPTER.decode(File(file).readBytes())
            return Model(modelScheme)
        }
    }
}
