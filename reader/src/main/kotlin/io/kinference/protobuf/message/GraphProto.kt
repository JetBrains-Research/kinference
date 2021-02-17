package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class GraphProto(
    //ProtoTag = 1
    val node: MutableList<NodeProto> = ArrayList(),

    //ProtoTag = 2
    var name: String? = null,

    //ProtoTag = 5
    val initializer: MutableList<TensorProto> = ArrayList(),

    //ProtoTag = 15
    val sparse_initializer: MutableList<SparseTensorProto> = ArrayList(),

    //ProtoTag = 10
    var doc_string: String? = null,

    //ProtoTag = 11
    val input: MutableList<ValueInfoProto> = ArrayList(),

    //ProtoTag = 12
    val output: MutableList<ValueInfoProto> = ArrayList(),

    //ProtoTag = 13
    val value_info: MutableList<ValueInfoProto> = ArrayList(),

    //ProtoTag = 14
    val quantization_annotation: MutableList<TensorAnnotation> = ArrayList(),
) {
    companion object {
        fun decode(reader: ProtobufReader): GraphProto {
            val proto = GraphProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.node.add(NodeProto.decode(reader))
                    2 -> proto.name = reader.readString()
                    5 -> proto.initializer.add(TensorProto.decode(reader))
                    15 -> proto.sparse_initializer.add(SparseTensorProto.decode(reader))
                    10 -> proto.doc_string = reader.readString()
                    11 -> proto.input.add(ValueInfoProto.decode(reader))
                    12 -> proto.output.add(ValueInfoProto.decode(reader))
                    13 -> proto.value_info.add(ValueInfoProto.decode(reader))
                    14 -> proto.quantization_annotation.add(TensorAnnotation.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
