package io.kinference.utils

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.map.ONNXMap
import io.kinference.data.seq.ONNXSequence
import io.kinference.data.tensors.Tensor
import io.kinference.protobuf.message.*

object DataLoader {
    fun getData(bytes: ByteArray, type: ONNXDataType): ONNXData = when (type) {
        ONNXDataType.ONNX_TENSOR -> getTensor(bytes)
        ONNXDataType.ONNX_SEQUENCE -> getSequence(bytes)
        ONNXDataType.ONNX_MAP -> getMap(bytes)
    }

    fun getTensor(bytes: ByteArray): Tensor = getTensor(TensorProto.decode(bytes))
    fun getSequence(bytes: ByteArray): ONNXSequence = getSequence(SequenceProto.decode(bytes))
    fun getMap(bytes: ByteArray): ONNXMap = getMap(MapProto.decode(bytes))

    fun getTensor(tensorProto: TensorProto) = Tensor.create(tensorProto)
    fun getSequence(seqProto: SequenceProto) = ONNXSequence.create(seqProto)
    fun getMap(mapProto: MapProto) = ONNXMap.create(mapProto)
}
