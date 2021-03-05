package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class OperatorSetIdProto(
    val domain: String? = null,
    val version: Long? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): OperatorSetIdProto {
            var domain: String? = null
            var version: Long? = null
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.DOMAIN -> domain = reader.readString()
                    ReaderTag.VERSION -> version = reader.readLong()
                }
            }
            return OperatorSetIdProto(domain, version)
        }
    }

    private enum class ReaderTag(val tag: Int) {
        DOMAIN(1),
        VERSION(2);

        companion object {
            fun fromInt(tag: Int) = values().first { it.tag == tag }
        }
    }
}
