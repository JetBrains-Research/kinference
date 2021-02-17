package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class OperatorSetIdProto(
    //ProtoTag = 1
    val domain: String? = null,

    //ProtoTag = 2
    val version: Long? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): OperatorSetIdProto {
            var domain: String? = null
            var version: Long? = null
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> domain = reader.readString()
                    2 -> version = reader.readLong()
                    else -> reader.readUnknownField(tag)
                }
            }
            return OperatorSetIdProto(domain = domain, version = version)
        }
    }
}
