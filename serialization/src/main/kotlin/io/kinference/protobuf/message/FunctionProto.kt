package io.kinference.protobuf.message

import com.squareup.wire.FieldEncoding
import com.squareup.wire.ProtoAdapter
import io.kinference.protobuf.ProtobufReader

class FunctionProto(
    var name: String? = null,
    var sinceVersion: Long? = null,
    var status: OperatorStatus? = null,
    val input: MutableList<String> = ArrayList(),
    val output: MutableList<String> = ArrayList(),
    val attribute: MutableList<String> = ArrayList(),
    val node: MutableList<NodeProto> = ArrayList(),
    val opSetImport: MutableList<OperatorSetIdProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): FunctionProto {
            val proto = FunctionProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.name = reader.readString()
                    2 -> proto.sinceVersion = reader.readLong()
                    3 -> try {
                        proto.status = reader.readValue(OperatorStatus.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    4 -> proto.input.add(reader.readString())
                    5 -> proto.output.add(reader.readString())
                    6 -> proto.attribute.add(reader.readString())
                    7 -> proto.node.add(NodeProto.decode(reader))
                    8 -> reader.readString() // skip docstring
                    9 -> proto.opSetImport.add(OperatorSetIdProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
