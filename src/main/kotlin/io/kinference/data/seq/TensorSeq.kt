package io.kinference.data.seq

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.tensors.Tensor
import io.kinference.types.SequenceInfo

class TensorSeq(val data: List<Tensor>, info: SequenceInfo) : ONNXData(ONNXDataType.ONNX_SEQUENCE, info) {
    override fun rename(newName: String): ONNXData = TensorSeq(data, SequenceInfo(newName, info.type))

    val length: Int
        get() = data.size
}
