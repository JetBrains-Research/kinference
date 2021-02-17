package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.FloatArrayDeserializer
import io.kinference.protobuf.arrays.LongArrayDeserializer
import okio.Buffer
import okio.ByteString

class AttributeProto(
    //ProtoTag = 1
    var name: String? = null,

    //ProtoTag = 21
    var ref_attr_name: String? = null,

    //ProtoTag = 13
    var doc_string: String? = null,

    //ProtoTag = 20
    var type: AttributeType? = null,

    //ProtoTag = 2
    var f: Float? = null,

    //ProtoTag = 3
    var i: Long? = null,

    //ProtoTag = 4
    var s: ByteString? = null,

    //ProtoTag = 5
    var t: TensorProto? = null,

    //ProtoTag = 6
    var g: GraphProto? = null,

    //ProtoTag = 22
    var sparse_tensor: SparseTensorProto? = null,

    //ProtoTag = 7
    var floats: FloatArray? = null,

    //ProtoTag = 8
    var ints: LongArray? = null,

    //ProtoTag = 9
    val strings: MutableList<ByteString> = ArrayList(),

    //ProtoTag = 10
    val tensors: MutableList<TensorProto> = ArrayList(),

    //ProtoTag = 11
    val graphs: MutableList<GraphProto> = ArrayList(),

    //ProtoTag = 23
    val sparse_tensors: MutableList<SparseTensorProto> = ArrayList(),
) {
    companion object {
        fun decode(reader: ProtobufReader): AttributeProto {
            val proto = AttributeProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.name = reader.readString()
                    21 -> proto.ref_attr_name = reader.readString()
                    13 -> proto.doc_string = reader.readString()
                    20 -> try {
                        proto.type = reader.readValue(AttributeType.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    2 -> proto.f = reader.readFloat()
                    3 -> proto.i = reader.readLong()
                    4 -> proto.s = reader.readBytes()
                    5 -> proto.t = TensorProto.decode(reader)
                    6 -> proto.g = GraphProto.decode(reader)
                    22 -> proto.sparse_tensor = SparseTensorProto.decode(reader)
                    7 -> proto.floats = FloatArrayDeserializer.decode(reader, tag)
                    8 -> proto.ints = LongArrayDeserializer.decode(reader, tag)
                    9 -> proto.strings.add(reader.readBytes())
                    10 -> proto.tensors.add(TensorProto.decode(reader))
                    11 -> proto.graphs.add(GraphProto.decode(reader))
                    23 -> proto.sparse_tensors.add(SparseTensorProto.decode(reader))
                    else -> {
                        reader.readUnknownField(tag)
                        null
                    }
                }
            }
            return proto
        }

        fun decode(bytes: ByteArray) = decode(ProtobufReader(Buffer().write(bytes)))
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
            val ADAPTER: ProtoAdapter<AttributeType> = object : EnumAdapter<AttributeType>(AttributeType::class) {
                override fun fromValue(value: Int): AttributeType? = AttributeType.fromValue(value)
            }

            fun fromValue(value: Int): AttributeType? = when (value) {
                0 -> UNDEFINED
                1 -> FLOAT
                2 -> INT
                3 -> STRING
                4 -> TENSOR
                5 -> GRAPH
                11 -> SPARSE_TENSOR
                6 -> FLOATS
                7 -> INTS
                8 -> STRINGS
                9 -> TENSORS
                10 -> GRAPHS
                12 -> SPARSE_TENSORS
                else -> null
            }
        }
    }
}
