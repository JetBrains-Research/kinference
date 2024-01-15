package io.kinference.tfjs.optimizer.rules

import io.kinference.tfjs.optimizer.rules.context.*

object OptimizerRuleSet {
    val DEFAULT_OPT_RULES = listOf(
        GRUContextRule,
        LSTMContextRule,
        ConvContextRule
    )
}
