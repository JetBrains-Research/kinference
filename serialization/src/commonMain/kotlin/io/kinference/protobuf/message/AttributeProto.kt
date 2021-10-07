package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.protobuf.*
import okio.Buffer

class AttributeProto(
    var name: String? = null,
    var refAttrName: String? = null,
    var type: AttributeType = AttributeType.UNDEFINED,
    var f: Float? = null,
    var i: Long? = null,
    var s: String? = null,
    var t: TensorProto? = null,
    var g: GraphProto? = null,
    var sparseTensor: SparseTensorProto? = null,
    var floats: FloatArray? = null,
    var ints: LongArray? = null,
    val strings: MutableList<String> = ArrayList(),
    val tensors: MutableList<TensorProto> = ArrayList(),
    val graphs: MutableList<GraphProto> = ArrayList(),
    val sparseTensors: MutableList<SparseTensorProto> = ArrayList()
) {
    companion object {
        fun decode(reader: ProtobufReader): AttributeProto {
            val proto = AttributeProto()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.NAME -> proto.name = reader.readString()
                    ReaderTag.REF_ATTR_NAME -> proto.refAttrName = reader.readString()
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                    ReaderTag.TYPE -> try {
                        proto.type = reader.readValue(AttributeType.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    ReaderTag.FLOAT_DATA -> proto.f = reader.readFloat()
                    ReaderTag.INT_DATA -> proto.i = reader.readLong()
                    ReaderTag.STRING_DATA -> proto.s = reader.readBytes().utf8()
                    ReaderTag.TENSOR_DATA -> proto.t = TensorProto.decode(reader)
                    ReaderTag.GRAPH_DATA -> proto.g = GraphProto.decode(reader)
                    ReaderTag.SPARSE_TENSOR -> proto.sparseTensor = SparseTensorProto.decode(reader)
                    ReaderTag.FLOATS_DATA -> proto.floats = reader.readFloatArray(tag)
                    ReaderTag.INTS_DATA -> proto.ints = reader.readLongArray(tag)
                    ReaderTag.STRINGS_DATA -> proto.strings.add(reader.readBytes().utf8())
                    ReaderTag.TENSORS_DATA -> proto.tensors.add(TensorProto.decode(reader))
                    ReaderTag.GRAPHS_DATA -> proto.graphs.add(GraphProto.decode(reader))
                    ReaderTag.SPARSE_TENSORS -> proto.sparseTensors.add(SparseTensorProto.decode(reader))
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        NAME(1),
        FLOAT_DATA(2),
        INT_DATA(3),
        STRING_DATA(4),
        TENSOR_DATA(5),
        GRAPH_DATA(6),
        FLOATS_DATA(7),
        INTS_DATA(8),
        STRINGS_DATA(9),
        TENSORS_DATA(10),
        GRAPHS_DATA(11),
        DOC_STRING(13),
        TYPE(20),
        REF_ATTR_NAME(21),
        SPARSE_TENSOR(22),
        SPARSE_TENSORS(23);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }

    enum class AttributeType(override val value: Int) : WireEnum {
        UNDEFINED(0),
        FLOAT(1),
        INT(2),
        STRING(3),
        TENSOR(4),
        GRAPH(5),
        SPARSE_TENSOR(11),
        FLOATS(6),
        INTS(7),
        STRINGS(8),
        TENSORS(9),
        GRAPHS(10),
        SPARSE_TENSORS(12);

        companion object {
            val ADAPTER: ProtoAdapter<AttributeType> = object : EnumAdapter<AttributeType>(AttributeType::class, Syntax.PROTO_2, UNDEFINED) {
                override fun fromValue(value: Int): AttributeType = AttributeType.fromValue(value)
            }

            fun fromValue(value: Int): AttributeType = when (value) {
                0 -> UNDEFINED
                1 -> FLOAT
                2 -> INT
                3 -> STRING
                4 -> TENSOR
                5 -> GRAPH
                6 -> FLOATS
                7 -> INTS
                8 -> STRINGS
                9 -> TENSORS
                10 -> GRAPHS
                11 -> SPARSE_TENSOR
                12 -> SPARSE_TENSORS
                else -> error("Cannot convert from value $value")
            }
        }
    }
}
