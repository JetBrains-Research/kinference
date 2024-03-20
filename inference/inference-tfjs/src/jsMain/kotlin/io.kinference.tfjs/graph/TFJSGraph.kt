package io.kinference.tfjs.graph

import io.kinference.graph.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.profiler.ProfilingContext
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.operators.TFJSOperatorFactory
import io.kinference.utils.closeAll

class TFJSGraph private constructor(
    proto: GraphProto,
    initializers: ArrayList<TFJSData<*>>,
    operators: ArrayList<Operator<TFJSData<*>, TFJSData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
    private val preparedTensorsContext: GraphContext<TFJSData<*>> = GraphContext()
) : Graph<TFJSData<*>>(proto, initializers, operators, valueOrderInfo) {

    override suspend fun close() {
        preparedTensorsContext.close()
        super.close()
    }

    override suspend fun applyWithAllocationControl(
        contexts: Contexts<TFJSData<*>>,
        profilingContext: ProfilingContext?,
        operator: Operator<TFJSData<*>, TFJSData<*>>
    ): List<TFJSData<*>?> {
        return operator.applyWithCheck(
            Contexts(contexts.graph, profilingContext),
            operator.inputs.map { input -> if (input.isEmpty()) null else contexts.graph!!.getValue(input) })
    }

    override suspend fun returnOutputsWithAllocationControl(contexts: Contexts<TFJSData<*>>): List<TFJSData<*>> {
        return outputs.map { contexts.graph!!.getValue(it.name) }
    }

    fun addTensorToContext(tensor: TFJSTensor) {
        preparedTensorsContext.putValue(tensor.name!!, tensor)
    }

    override fun makeContext(root: GraphContext<TFJSData<*>>?): GraphContext<TFJSData<*>> {
        val context = GraphContext(root)
        context.mergeContext(preparedTensorsContext)
        return context
    }

    companion object: CompanionInitializers<TFJSData<*>> {
        suspend operator fun invoke(proto: GraphProto, opSetRegistry: OperatorSetRegistry): TFJSGraph {
            val valueOrderInfo = GraphValueOrderInfo()
            val nodes = proto.collectOperators<TFJSData<*>>(valueOrderInfo)
            val operators = ArrayList<Operator<TFJSData<*>, TFJSData<*>>>(nodes.size).apply {
                try {
                    for (node in nodes) {
                        add(TFJSOperatorFactory.create(node.proto, opSetRegistry))
                    }
                } catch (e: Exception) {
                    closeAll(this)
                    throw e
                }
            }

            val initializers = getInitializers(proto)
            return TFJSGraph(proto, initializers, operators, valueOrderInfo)
        }

        private suspend fun getInitializers(proto: GraphProto): ArrayList<TFJSData<*>> {
            val initializers = ArrayList<TFJSData<*>>(proto.initializer.size).apply {
                for (i in proto.initializer)
                    this.add(prepareInput(i))
            }
            return initializers
        }

        override suspend fun prepareInput(proto: TensorProto): TFJSData<*> = TFJSTensor.create(proto)
    }
}
