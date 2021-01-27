package io.kinference.algorithms.gec

/**
 * Tags used by GEC model
 */
enum class GECTag(val value: String, val isNonChanging: Boolean = false) {
    KEEP("\$KEEP", isNonChanging = true),
    PAD("@@PADDING@@", isNonChanging = true),
    UNK("@@UNKNOWN@@", isNonChanging = true),
    CORRECT("CORRECT"),
    INCORRECT("INCORRECT"),
    DELETE("\$DELETE");

    companion object {
        fun from(value: String) = values().singleOrNull { it.value == value }
    }
}

