package io.kinference.operator

import io.kinference.protobuf.message.OperatorSetIdProto

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
