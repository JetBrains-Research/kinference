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
        fun decode(reader: ProtobufReader): OperatorProto {
            val proto = OperatorProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.opType = reader.readString()
                    2 -> proto.sinceVersion = reader.readLong()
                    3 -> try {
                        proto.status = reader.readValue(OperatorStatus.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    10 -> reader.readString() // skip docstring
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
