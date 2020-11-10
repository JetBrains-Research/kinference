package io.kinference.data.seq

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.tensors.Tensor
import io.kinference.types.SequenceInfo

class TensorSeq(val data: List<Tensor>, info: SequenceInfo) : ONNXData(ONNXDataType.ONNX_SEQUENCE, info) {
    override fun rename(name: String): ONNXData = TensorSeq(data, SequenceInfo(name, info.type))

    val length: Int = data.size
}
