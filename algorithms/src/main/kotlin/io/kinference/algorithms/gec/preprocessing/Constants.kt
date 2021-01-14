package io.kinference.algorithms.gec.preprocessing

/**
 * abstract class-constant for Tag
 */
sealed class Tag {
    abstract val value: String

    /**
     * Tag which represents KEEP tag
     */
    object Keep : Tag() {
        override val value: String = "\$KEEP"
    }

    /**
     * Tag which represents PAD tag
     */
    object Pad : Tag() {
        override val value: String = "@@PADDING@@"
    }

    /**
     * Tag which represents UNKNOWN tag
     */
    object Unk : Tag() {
        override val value: String = "@@UNKNOWN@@"
    }

    /**
     * Tag which represents CORRECT tag
     */
    object Correct : Tag() {
        override val value: String = "CORRECT"
    }

    /**
     * Tag which represents INCORRECT tag
     */
    object Incorrect : Tag() {
        override val value: String = "INCORRECT"
    }

    /**
     * Tag which represents DELETE tag
     */
    object Delete : Tag() {
        override val value: String = "\$DELETE"
    }
}

