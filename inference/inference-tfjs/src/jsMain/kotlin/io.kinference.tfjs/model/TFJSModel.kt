package io.kinference.tfjs.model

import io.kinference.data.ONNXData
import io.kinference.model.Model
import io.kinference.protobuf.message.ModelProto
import io.kinference.tfjs.graph.Graph
import io.kinference.tfjs.utils.setDefaultBackend
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TFJSModel(proto: ModelProto) : Model<ONNXData<*>> {
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

    override fun predict(input: Map<String, ONNXData<*>>, profile: Boolean): Map<String, ONNXData<*>> {
        val inputs = input.values
        return graph.execute(inputs).associateBy { it.name.orEmpty() }
    }
}
