package io.kinference.data.map

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.tensors.Tensor
import io.kinference.types.ValueInfo

class Map(val data: List<Tensor>, info: ValueInfo.MapInfo) : ONNXData(ONNXDataType.ONNX_MAP, info) {
    override fun rename(name: String): ONNXData {
        TODO("Not yet implemented")
    }
}
