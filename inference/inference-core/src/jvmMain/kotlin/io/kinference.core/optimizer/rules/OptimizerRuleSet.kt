package io.kinference.core.optimizer.rules

import io.kinference.core.optimizer.rules.context.*

object OptimizerRuleSet {
    val DEFAULT_OPT_RULES = listOf(
        AttentionContextRule,
        QAttentionContextRule,
        GRUContextRule,
        LSTMContextRule,
        DynamicQuantizeLSTMContextRule,
        MatMulIntegerContextRule,
    )
}
