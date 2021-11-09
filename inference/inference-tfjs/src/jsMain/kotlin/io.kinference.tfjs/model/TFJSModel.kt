package io.kinference.tfjs.model

import io.kinference.model.Model
import io.kinference.protobuf.message.ModelProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.graph.Graph
import io.kinference.tfjs.utils.setDefaultBackend
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TFJSModel(proto: ModelProto) : Model<TFJSData<*>> {
    init {
        setDefaultBackend()
    }

    val graph = Graph(proto.graph!!)
    val name: String = "${proto.domain}:${proto.modelVersion}"

   /* companion object {
        fun load(bytes: ByteArray): TFJSModel {
            val modelScheme = ModelProto.decode(bytes)
            return TFJSModel(modelScheme)
        }
    }*/

    override fun predict(input: List<TFJSData<*>>, profile: Boolean): Map<String, TFJSData<*>> {
        return graph.execute(input).associateBy { it.name.orEmpty() }
    }
}
