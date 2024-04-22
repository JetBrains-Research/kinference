package io.kinference.protobuf.message

import com.squareup.wire.FieldEncoding
import com.squareup.wire.ProtoAdapter
import io.kinference.protobuf.ProtobufReader

class OperatorProto(
    var opType: String? = null,
    var sinceVersion: Long? = null,
    var status: OperatorStatus? = null
) {
    companion object {
        suspend fun decode(reader: ProtobufReader): OperatorProto {
            val proto = OperatorProto()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.OP_TYPE -> proto.opType = reader.readString()
                    ReaderTag.SINCE_VERSION -> proto.sinceVersion = reader.readLong()
                    ReaderTag.STATUS -> try {
                        proto.status = reader.readValue(OperatorStatus.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        OP_TYPE(1),
        SINCE_VERSION(2),
        STATUS(3),
        DOC_STRING(10);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
