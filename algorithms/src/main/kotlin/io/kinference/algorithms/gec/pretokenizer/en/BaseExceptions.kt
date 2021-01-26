package io.kinference.algorithms.gec.pretokenizer.en

/**
 * Basic Exceptions for English (not only) tokenizer
 */
class BaseExceptions {

    private val spaces = listOf(" ", "\t", "\\t", "\n", "\\n", "\u2014", "\u00a0")

    private val ends = listOf("'",  """\\")""", "<space>", "''", "C++", "a.", "b.", "c.", "d.", "e.", "f.", "g.", "h.", "i.",
        "j.", "k.", "l.", "m.", "n.", "o.", "p.", "q.", "r.", "s.", "t.", "u.", "v.", "w.", "x.", "y.", "z.", "ä.", "ö.", "ü.",)

    private val emoticons = """:) :-) :)) :-)) :))) :-))) (: (-: =) (= ") :] :-] [: [-: [= =] :o) (o: :} :-} 
        8) 8-) (-8 ;) ;-) (; (-; :( :-( :(( :-(( :((( :-((( ): )-: =( >:( :') :'-) :'( :'-( :/ :-/ =/ =| :| :-| ]= 
        =[ :1 :P :-P :p :-p :O :-O :o :-o :0 :-0 :() >:o :* :-* :3 :-3 =3 :> :-> :X :-X :x :-x :D :-D ;D ;-D =D xD 
        XD xDD XDD 8D 8-D ^_^ ^__^ ^___^ >.< >.> <.< ._. ;_; -_- -__- v.v V.V v_v V_V o_o o_O O_o O_O 0_o o_0 0_0 o.O 
        O.o O.O o.o 0.0 o.0 0.o @_@ <3 <33 <333 </3 (^_^) (-_-) (._.) (>_<) (*_*) (¬_¬) ಠ_ಠ ಠ︵ಠ (ಠ_ಠ) ¯\(ツ)/¯ (╯°□°）╯︵┻━┻ ><(((*>""".split(" ")

    val exceptions = (spaces + ends + emoticons).map { it to listOf(TokenInfo(orth = it)) }.toMap() as HashMap<String, List<TokenInfo>>
}
