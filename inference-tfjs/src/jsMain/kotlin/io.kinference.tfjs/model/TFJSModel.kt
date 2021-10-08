package io.kinference.tfjs.model

import io.kinference.data.ONNXData
import io.kinference.model.Model
import io.kinference.protobuf.message.ModelProto
import io.kinference.tfjs.data.TFJSData
import io.kinference.tfjs.graph.Graph
import io.kinference.tfjs.utils.setDefaultBackend
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TFJSModel(proto: ModelProto) : Model {
    init {
        setDefaultBackend()
    }

    val graph = Graph(proto.graph!!)
    val name: String = "${proto.domain}:${proto.modelVersion}"

    companion object {
        fun load(bytes: ByteArray): TFJSModel {
            val modelScheme = ModelProto.decode(bytes)
            return TFJSModel(modelScheme)
        }
    }

    override fun predict(input: List<ONNXData<*>>, profile: Boolean): List<TFJSData<*>> {
        return graph.execute(input as List<TFJSData<*>>)
    }
}
