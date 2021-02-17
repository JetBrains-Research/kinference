package io.kinference.protobuf.message

import com.squareup.wire.FieldEncoding
import com.squareup.wire.ProtoAdapter
import io.kinference.protobuf.ProtobufReader

class OperatorProto(
    //ProtoTag = 1
    var op_type: String? = null,

    //ProtoTag = 2
    var since_version: Long? = null,

    //ProtoTag = 3
    var status: OperatorStatus? = null,

    //ProtoTag = 10
    var doc_string: String? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): OperatorProto {
            val proto = OperatorProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.op_type = reader.readString()
                    2 -> proto.since_version = reader.readLong()
                    3 -> try {
                        proto.status = reader.readValue(OperatorStatus.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    10 -> proto.doc_string = reader.readString()
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
