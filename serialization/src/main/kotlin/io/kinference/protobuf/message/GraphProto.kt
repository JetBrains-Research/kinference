package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class GraphProto(
    val node: MutableList<NodeProto> = ArrayList(),
    var name: String? = null,
    val initializer: MutableList<TensorProto> = ArrayList(),
    val sparseInitializer: MutableList<SparseTensorProto> = ArrayList(),
    val input: MutableList<ValueInfoProto> = ArrayList(),
    val output: MutableList<ValueInfoProto> = ArrayList(),
    val valueInfo: MutableList<ValueInfoProto> = ArrayList(),
    val quantizationAnnotation: MutableList<TensorAnnotation> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): GraphProto {
            val proto = GraphProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.node.add(NodeProto.decode(reader))
                    2 -> proto.name = reader.readString()
                    5 -> proto.initializer.add(TensorProto.decode(reader))
                    10 -> reader.readString() // skip docstring
                    11 -> proto.input.add(ValueInfoProto.decode(reader))
                    12 -> proto.output.add(ValueInfoProto.decode(reader))
                    13 -> proto.valueInfo.add(ValueInfoProto.decode(reader))
                    14 -> proto.quantizationAnnotation.add(TensorAnnotation.decode(reader))
                    15 -> proto.sparseInitializer.add(SparseTensorProto.decode(reader))
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }
}
