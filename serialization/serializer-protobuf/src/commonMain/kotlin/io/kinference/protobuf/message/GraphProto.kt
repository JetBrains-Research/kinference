package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.readTensor

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
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.NODE -> proto.node.add(NodeProto.decode(reader))
                    ReaderTag.NAME -> proto.name = reader.readString()
                    ReaderTag.INITIALIZER -> proto.initializer.add(reader.readTensor())
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                    ReaderTag.INPUT -> proto.input.add(ValueInfoProto.decode(reader))
                    ReaderTag.OUTPUT -> proto.output.add(ValueInfoProto.decode(reader))
                    ReaderTag.VALUE_INFO -> proto.valueInfo.add(ValueInfoProto.decode(reader))
                    ReaderTag.QUANTIZATION_ANNOTATION -> proto.quantizationAnnotation.add(TensorAnnotation.decode(reader))
                    ReaderTag.SPARSE_INITIALIZER -> proto.sparseInitializer.add(SparseTensorProto.decode(reader))
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        NODE(1),
        NAME(2),
        INITIALIZER(5),
        DOC_STRING(10),
        INPUT(11),
        OUTPUT(12),
        VALUE_INFO(13),
        QUANTIZATION_ANNOTATION(14),
        SPARSE_INITIALIZER(15);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
