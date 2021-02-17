package io.kinference.protobuf.message

import com.squareup.wire.FieldEncoding
import com.squareup.wire.ProtoAdapter
import io.kinference.protobuf.ProtobufReader

class FunctionProto(
    //ProtoTag = 1
    var name: String? = null,

    //ProtoTag = 2
    var since_version: Long? = null,

    //ProtoTag = 3
    var status: OperatorStatus? = null,

    //ProtoTag = 4
    val input: MutableList<String> = ArrayList(),

    //ProtoTag = 5
    val output: MutableList<String> = ArrayList(),

    //ProtoTag = 6
    val attribute: MutableList<String> = ArrayList(),

    //ProtoTag = 7
    val node: MutableList<NodeProto> = ArrayList(),

    //ProtoTag = 8
    var doc_string: String? = null,

    //ProtoTag = 9
    val opset_import: MutableList<OperatorSetIdProto> = ArrayList(),
) {

    companion object {
        fun decode(reader: ProtobufReader): FunctionProto {
            val proto = FunctionProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.name = reader.readString()
                    2 -> proto.since_version = reader.readLong()
                    3 -> try {
                        proto.status = reader.readValue(OperatorStatus.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    4 -> proto.input.add(reader.readString())
                    5 -> proto.output.add(reader.readString())
                    6 -> proto.attribute.add(reader.readString())
                    7 -> proto.node.add(NodeProto.decode(reader))
                    8 -> proto.doc_string = reader.readString()
                    9 -> proto.opset_import.add(OperatorSetIdProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
