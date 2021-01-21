package io.kinference.algorithms.gec.changes

/** Defines tokens that should be appended to text without space on the left */
internal object AppendNoSpace {
    private val values: Set<String> = setOf("-", ",", ".", ";", ":", "!", "?", "'s", "'m", "'d", "'ve", "'ll", "'re", "n't", "'t")

    fun inNoSpaceCharacters(tok: String): Boolean {
        return tok in values
    }
}
