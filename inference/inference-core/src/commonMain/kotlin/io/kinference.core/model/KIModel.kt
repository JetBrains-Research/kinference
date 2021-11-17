package io.kinference.core.model

import io.kinference.core.KIONNXData
import io.kinference.core.graph.*
import io.kinference.core.operators.OperatorInfo
import io.kinference.model.Model
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto
import io.kinference.protobuf.message.OperatorSetIdProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class KIModel(proto: ModelProto) : Model<KIONNXData<*>>, Profilable {
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

    val name: String = "${proto.domain}:${proto.modelVersion}"
    private val opSet = OperatorSetRegistry(proto.opSetImport)
    val graph = Graph(proto.graph!!, opSet)

    private val profiles: MutableList<ProfilingContext> = ArrayList()
    override fun addContext(name: String): ProfilingContext = ProfilingContext(name).apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override fun predict(input: List<KIONNXData<*>>, profile: Boolean): Map<String, KIONNXData<*>> {
        val context = if (profile) addContext("Model $name") else null
        val execResult = graph.execute(input, profilingContext = context)
        return execResult.associateBy { it.name!! }
    }
}
