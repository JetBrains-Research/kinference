package io.kinference.core.graph

import io.kinference.core.*
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.operators.KIOperatorFactory
import io.kinference.graph.*
import io.kinference.ndarray.arrays.ArrayUsageMarker
import io.kinference.ndarray.arrays.ArrayDispatcher
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto


class KIGraph private constructor(
    proto: GraphProto,
    operators: ArrayList<Operator<KIONNXData<*>, KIONNXData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
    private val preparedTensorsContext: GraphContext<KIONNXData<*>> = GraphContext()
) : Graph<KIONNXData<*>>(proto, operators, valueOrderInfo) {
    override fun makeContext(root: GraphContext<KIONNXData<*>>?): GraphContext<KIONNXData<*>> {
        val context = GraphContext(root)
        context.mergeContext(preparedTensorsContext)
        return context
    }

    override fun prepareInput(proto: TensorProto): KIONNXData<*> = KITensor.create(proto)

    override suspend fun applyWithAllocationControl(
        contexts: Contexts<KIONNXData<*>>,
        profilingContext: ProfilingContext?,
        operator: Operator<KIONNXData<*>, KIONNXData<*>>
    ): List<KIONNXData<*>?> {
        ArrayDispatcher.setOperatorContext(operator.operatorIndex)
        val outputs = operator.applyWithCheck(
            Contexts(contexts.graph, profilingContext),
            operator.inputs.map { input -> if (input.isEmpty()) null else contexts.graph!!.getValue(input) })
        outputs.forEach { it?.markOutput(ArrayUsageMarker.ContextOutput) }
        ArrayDispatcher.releaseUsedInContext()

        return outputs
    }

    override suspend fun returnOutputsWithAllocationControl(contexts: Contexts<KIONNXData<*>>): List<KIONNXData<*>> {
        val result = outputs.map { contexts.graph!!.getValue(it.name) }
        result.forEach { it.markOutput(ArrayUsageMarker.GlobalOutput) }
        ArrayDispatcher.releaseAllOutputArrays()
        return result
    }

    fun addTensorToContext(tensor: KITensor) {
        preparedTensorsContext.putValue(tensor.name!!, tensor)
    }

    companion object {
        suspend operator fun invoke(proto: GraphProto, opSetRegistry: OperatorSetRegistry): KIGraph {
            val valueOrderInfo = GraphValueOrderInfo()
            val nodes = proto.collectOperators<KIONNXData<*>>(valueOrderInfo)
            val operators = ArrayList<Operator<KIONNXData<*>, KIONNXData<*>>>(nodes.size).apply {
                for (node in nodes) {
                    add(KIOperatorFactory.create(node.proto, opSetRegistry))
                }
            }

            return KIGraph(proto, operators, valueOrderInfo)
        }
    }
}
