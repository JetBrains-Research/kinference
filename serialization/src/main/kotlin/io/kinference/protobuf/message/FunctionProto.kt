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
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.NAME -> proto.name = reader.readString()
                    ReaderTag.SINCE_VERSION -> proto.sinceVersion = reader.readLong()
                    ReaderTag.STATUS -> try {
                        proto.status = reader.readValue(OperatorStatus.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    ReaderTag.INPUT -> proto.input.add(reader.readString())
                    ReaderTag.OUTPUT -> proto.output.add(reader.readString())
                    ReaderTag.ATTRIBUTE -> proto.attribute.add(reader.readString())
                    ReaderTag.NODE -> proto.node.add(NodeProto.decode(reader))
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                    ReaderTag.OP_SET_IMPORT -> proto.opSetImport.add(OperatorSetIdProto.decode(reader))
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        NAME(1),
        SINCE_VERSION(2),
        STATUS(3),
        INPUT(4),
        OUTPUT(5),
        ATTRIBUTE(6),
        NODE(7),
        DOC_STRING(8),
        OP_SET_IMPORT(9);

        companion object {
            fun fromInt(tag: Int) = values().first { it.tag == tag }
        }
    }
}
