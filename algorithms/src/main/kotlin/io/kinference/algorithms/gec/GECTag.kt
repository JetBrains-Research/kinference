package io.kinference.algorithms.gec

/**
 * Tags used by GEC model
 */
enum class GECTag(val value: String) {
    KEEP("\$KEEP"),
    PAD("@@PADDING@@"),
    UNK("@@UNKNOWN@@"),
    CORRECT("CORRECT"),
    INCORRECT("INCORRECT"),
    DELETE("\$DELETE")
}

