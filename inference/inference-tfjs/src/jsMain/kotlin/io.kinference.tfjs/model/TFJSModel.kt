package io.kinference.tfjs.model

import io.kinference.model.Model
import io.kinference.protobuf.message.ModelProto
import io.kinference.protobuf.message.OperatorSetIdProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.graph.Graph
import io.kinference.tfjs.operators.OperatorInfo
import io.kinference.tfjs.utils.setDefaultBackend
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TFJSModel(proto: ModelProto) : Model<TFJSData<*>> {
    init {
        setDefaultBackend()
    }

    class OperatorSetRegistry(proto: List<OperatorSetIdProto>) {
        private val operatorSets = HashMap<String, Int>().apply {
            for (opSet in proto) {
                val name = opSet.domain ?: OperatorInfo.DEFAULT_DOMAIN
                val version = opSet.version?.toInt() ?: 1
                put(name, version)
            }
        }

        fun getVersion(domain: String?): Int? {
            val domainName = domain ?: OperatorInfo.DEFAULT_DOMAIN
            return operatorSets[domainName]
        }
    }

    private val opSet = OperatorSetRegistry(proto.opSetImport)
    val graph = Graph(proto.graph!!, opSet)
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
