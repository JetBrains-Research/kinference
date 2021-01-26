package io.kinference.algorithms.gec.pretokenizer.en

/**
 * TokenizerExceptions class for generating/storing specific exception for tokenizer
 * @param exclude words to exlude from exceptions
 * @param exceptions every possible combination of pronoun/verb and a lot of specific words for right tokenization of this words
 */
class TokenizerExceptions {

    private val exclude = listOf("Ill", "ill", "Its", "its", "Hell",
        "hell", "Shell", "shell", "Shed", "shed", "were", "Were", "Well", "well", "Whore", "whore",)

    val exceptions = HashMap<String, List<TokenInfo>>()

    init {
        generateException()
    }

    private fun generateException(){
        for (pron in listOf("i")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth'm"] = listOf( TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'m", lemma = "be", norm = "am"))

                exceptions[orth + "m"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "m", lemma = "be"))

                exceptions["$orth'ma"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'m", lemma = "be", norm = "am"),
                    TokenInfo(orth = "a", lemma = "going to", norm = "gonna"))

                exceptions[orth + "ma"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "m", lemma = "be", norm = "am"),
                    TokenInfo(orth = "a", lemma = "going to", norm = "gonna"))

            }
        }

        for (pron in listOf("i", "you", "he", "she", "it", "we", "they")){
            for (orth in listOf(pron, pron.capitalize())){

                exceptions["$orth'll"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'ll", lemma = "will", norm = "will"))

                exceptions[orth + "ll"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "ll", lemma = "will", norm = "will"))

                exceptions["$orth'll've"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'ll", lemma = "will", norm = "will"),
                    TokenInfo(orth = "'ve", lemma = "have", norm = "have"))

                exceptions[orth + "llve"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "ll", norm = "will", lemma = "will"),
                    TokenInfo(orth = "ve", lemma = "have", norm = "have"))

                exceptions["$orth'd"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'d", norm = "'d"))

                exceptions[orth + "d"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "d", norm = "'d"))

                exceptions["$orth'd've"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'d", lemma = "would", norm = "would"),
                    TokenInfo(orth = "'ve", lemma = "have", norm = "have"))

                exceptions[orth + "dve"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "d", lemma = "would", norm = "would"),
                    TokenInfo(orth = "ve", lemma = "have", norm = "have"))
            }
        }

        for (pron in listOf("i", "you", "we", "they")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth've"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'ve", lemma = "have", norm = "have"))

                exceptions[orth + "ve"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "ve", lemma = "have", norm = "have"))
            }
        }

        for (pron in listOf("you", "we", "they")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth're"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'re", lemma = "be", norm = "are"))

                exceptions[orth + "re"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "re", lemma = "be", norm = "are"))
            }
        }

        for (pron in listOf("he", "she", "it")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth's"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "'s", norm = "'s"))

                exceptions[orth + "s"] = listOf(TokenInfo(orth = orth, norm = pron),
                    TokenInfo(orth = "s"))
            }
        }

        for (word in listOf("who", "what", "when", "where", "why", "how", "there", "that", "this", "these", "those")){
            for (orth in listOf(word, word.capitalize())){
                exceptions["$orth's"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word), TokenInfo(orth = "'s", norm = "'s"))

                exceptions[orth + "s"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word), TokenInfo(orth = "s"))

                exceptions["$orth'll"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "'ll", lemma = "will", norm = "will"))

                exceptions[orth + "ll"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "ll", lemma = "will", norm = "will"))

                exceptions["$orth'll've"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "'ll", lemma = "will", norm = "will"),
                    TokenInfo(orth = "'ve", lemma = "have", norm = "have"))

                exceptions[orth + "llve"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "ll", lemma = "will", norm = "will"),
                    TokenInfo(orth = "ve", lemma = "have", norm = "have"))

                exceptions["$orth're"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "'re", lemma = "be", norm = "are"))

                exceptions[orth + "re"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "re", lemma = "be", norm = "are"))

                exceptions["$orth've"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "'ve", lemma = "have"))

                exceptions[orth + "ve"] = listOf(TokenInfo(orth = orth, lemma = word),
                    TokenInfo(orth = "ve", lemma = "have", norm = "have"))

                exceptions["$orth'd"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                TokenInfo(orth = "'d", norm = "'d"))

                exceptions[orth + "d"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "d", norm = "'d"))

                exceptions["$orth'd've"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "'d", lemma = "would", norm = "would"),
                    TokenInfo(orth = "'ve", lemma = "have", norm = "have"))

                exceptions[orth + "dve"] = listOf(TokenInfo(orth = orth, lemma = word, norm = word),
                    TokenInfo(orth = "d", lemma = "would", norm = "would"),
                    TokenInfo(orth = "ve", lemma = "have", norm = "have"))
            }
        }

        val verbDatas = listOf(TokenInfo(orth = "ca", lemma = "can", norm = "can"),
            TokenInfo(orth = "do", lemma = "do", norm = "do"),
            TokenInfo(orth = "does", lemma = "do", norm = "does"),
            TokenInfo(orth = "did", lemma = "do", norm = "do"),
            TokenInfo(orth = "had", lemma = "have", norm = "have"),
            TokenInfo(orth = "may", norm = "may"),
            TokenInfo(orth = "need", norm = "need"),
            TokenInfo(orth = "ought", norm = "ought"),
            TokenInfo(orth = "sha", lemma = "shall", norm = "shall"),
            TokenInfo(orth = "wo", lemma = "will", norm = "will"))

        val verbs2 = listOf(TokenInfo(orth = "could", norm = "could"),
            TokenInfo(orth = "might", norm = "might"),
            TokenInfo(orth = "must", norm = "must"),
            TokenInfo(orth = "should", norm = "should"),
            TokenInfo(orth = "would", norm = "would"))

        for (verbData in verbDatas + verbs2){
            val verbDataC = verbData
            verbDataC.orth = verbDataC.orth.capitalize()
            for (data in listOf(verbData, verbDataC)){
                exceptions[data.orth + "n't"] = listOf(data, TokenInfo(orth = "n't", lemma = "not", norm = "not"))

                exceptions[data.orth + "nt"] = listOf(data, TokenInfo(orth = "nt", lemma = "not", norm = "not"))

                exceptions[data.orth + "n't've"] = listOf(data,
                    TokenInfo(orth = "n't", lemma = "not", norm = "not"),
                    TokenInfo(orth = "'ve", lemma = "have", norm = "have"))

                exceptions[data.orth + "ntve"] = listOf(data,
                    TokenInfo(orth = "nt", lemma = "not", norm = "not"),
                    TokenInfo(orth = "ve", lemma = "have", norm = "have"))
            }
        }
        for (verbData in verbs2){
            val verbDataC = verbData
            verbDataC.orth = verbDataC.orth.capitalize()
            for (data in listOf(verbData, verbDataC)){
                exceptions[data.orth + "'ve"] = listOf(data, TokenInfo(orth = "'ve", lemma = "have"))

                exceptions[data.orth + "ve"] = listOf(data, TokenInfo(orth = "ve", lemma = "have"))
            }
        }

        val beVariations = listOf(TokenInfo(orth = "ai", lemma = "be"),
            TokenInfo(orth = "are", lemma = "be", norm = "are"),
            TokenInfo(orth = "is", lemma = "be", norm = "are"),
            TokenInfo(orth = "was", lemma = "be", norm = "is"),
            TokenInfo(orth = "was", lemma = "be", norm = "was"),
            TokenInfo(orth = "were", lemma = "be", norm = "were"),
            TokenInfo(orth = "have", norm = "have"),
            TokenInfo(orth = "has", lemma = "have", norm = "has"),
            TokenInfo(orth = "dare", norm = "dare"))

        for (verbData in beVariations){
            val verbDataC = verbData
            verbDataC.orth = verbDataC.orth.capitalize()
            for (data in listOf(verbData, verbDataC)){
                exceptions[data.orth + "n't"] = listOf(data, TokenInfo(orth = "n't", lemma = "not", norm = "not"))

                exceptions[data.orth + "nt"] = listOf(data, TokenInfo(orth = "nt", lemma = "not", norm = "not"))
            }
        }

        val trailingApostrpopheData = listOf(TokenInfo(orth = "doin", lemma = "do", norm = "doing"),
            TokenInfo(orth = "goin", lemma = "go", norm = "going"),
            TokenInfo(orth = "nothin", lemma = "nothing", norm = "nothing"),
            TokenInfo(orth = "nuthin", lemma = "nothing", norm = "nothing"),
            TokenInfo(orth = "ol", lemma = "old", norm = "old"),
            TokenInfo(orth = "somethin", lemma = "something", norm = "something"))
        for (excData in trailingApostrpopheData){
            val excDataC = excData
            excDataC.orth = excDataC.orth.capitalize()
            for (data in listOf(excData)){
                val dataApos = data
                dataApos.orth = dataApos.orth + "'"
                exceptions[data.orth] = listOf(data)
                exceptions[dataApos.orth] = listOf(dataApos)
            }
        }

        val otherData = listOf(TokenInfo(orth = "em", norm = "them"),
            TokenInfo(orth = "ll", lemma = "will", norm = "will"),
            TokenInfo(orth = "nuff", lemma = "enough", norm = "enough"))

        for (excData in otherData){
            val excDataApos = excData
            excDataApos.orth = "'" + excDataApos.orth
            for (data in listOf(excData, excDataApos)){
                exceptions[data.orth] = listOf(data)
            }
        }

        // Times

        for (h in 1..12){
            for (period in listOf("a.m.", "am")){
                exceptions["$h$period"] = listOf(TokenInfo(orth = "$h"),
                    TokenInfo(orth = "$period", lemma = "a.m.", norm = "a.m."))
            }
            for (period in listOf("p.m.", "pm")){
                exceptions["$h$period"] = listOf(TokenInfo(orth = "$h"),
                    TokenInfo(orth = "$period", lemma = "a.m.", norm = "a.m."))
            }
        }

        val otherExc = hashMapOf("y'all" to listOf(TokenInfo(orth = "y'", norm = "you"), TokenInfo(orth = "all")),
                                 "yall" to listOf(TokenInfo(orth = "y", norm = "you"), TokenInfo(orth = "all")),
                                 "how'd'y" to listOf(TokenInfo(orth = "how", lemma = "how"), TokenInfo(orth = "'d", lemma = "do"), TokenInfo(orth = "'y", norm = "you")),
                                 "How'd'y" to listOf(TokenInfo(orth = "How", lemma = "how", norm = "how"), TokenInfo(orth = "'d", lemma = "do"), TokenInfo(orth = "'y", norm = "you")),
                                 "not've" to listOf(TokenInfo(orth = "not", lemma = "not"), TokenInfo(orth = "'ve", lemma = "have", norm = "have")),
                                 "notve" to listOf(TokenInfo(orth = "not", lemma = "not"), TokenInfo(orth = "ve", lemma = "have", norm = "have")),
                                 "Not've" to listOf(TokenInfo(orth = "Not", lemma = "not", norm = "not"), TokenInfo(orth = "'ve", lemma = "have", norm = "have")),
                                 "Notve" to listOf(TokenInfo(orth = "Not", lemma = "not", norm = "not"), TokenInfo(orth = "ve", lemma = "have", norm = "have")),
                                 "cannot" to listOf(TokenInfo(orth = "can", lemma = "can"), TokenInfo(orth = "not", lemma = "not")),
                                 "Cannot" to listOf(TokenInfo(orth = "Can", lemma = "can", norm = "can"), TokenInfo(orth = "not", lemma = "not")),
                                 "gonna" to listOf(TokenInfo(orth = "gon", lemma = "go", norm = "going"), TokenInfo(orth = "na", lemma = "to", norm = "to")),
                                 "Gonna" to listOf(TokenInfo(orth = "Gon", lemma = "go", norm = "going"), TokenInfo(orth = "na", lemma = "to", norm = "to")),
                                 "gotta" to listOf(TokenInfo(orth = "got"), TokenInfo(orth = "ta", lemma = "to", norm = "to")),
                                 "Gotta" to listOf(TokenInfo(orth = "Got", norm = "got"), TokenInfo(orth = "ta", lemma = "to", norm = "to")),
                                 "let's" to listOf(TokenInfo(orth = "let"), TokenInfo(orth = "'s", norm = "us")),
                                 "Let's" to listOf(TokenInfo(orth = "Let", lemma = "let", norm = "let"), TokenInfo(orth = "'s", norm = "us")),
                                 "c'mon" to listOf(TokenInfo(orth = "c'm", norm = "come", lemma = "come"), TokenInfo(orth = "on")),
                                 "C'mon" to listOf(TokenInfo(orth = "C'm", norm = "come", lemma = "come"), TokenInfo(orth = "on")))
        exceptions.putAll(otherExc)

        val abbreviations = listOf(TokenInfo(orth = "'S", lemma = "'s", norm = "'s"),
            TokenInfo(orth = "'s", lemma = "'s", norm = "'s"),
            TokenInfo(orth = "\\u2018S", lemma = "'S", norm = "'S"),
            TokenInfo(orth = "\\u2018s", lemma = "'s", norm = "'s"),
            TokenInfo(orth = "and/or", lemma = "and/or", norm = "and/or"),
            TokenInfo(orth = "w/o", lemma = "without", norm = "without"),
            TokenInfo(orth = "'re", lemma = "be", norm = "are"),
            TokenInfo(orth = "'Cause", lemma = "because", norm = "because"),
            TokenInfo(orth = "'cause", lemma = "because", norm = "because"),
            TokenInfo(orth = "'cos", lemma = "because", norm = "because"),
            TokenInfo(orth = "'Cos", lemma = "because", norm = "because"),
            TokenInfo(orth = "'coz", lemma = "because", norm = "because"),
            TokenInfo(orth = "'Coz", lemma = "because", norm = "because"),
            TokenInfo(orth = "'cuz", lemma = "because", norm = "because"),
            TokenInfo(orth = "'Cuz", lemma = "because", norm = "because"),
            TokenInfo(orth = "'bout", lemma = "about", norm = "about"),
            TokenInfo(orth = "ma'am", lemma = "madam", norm = "madam"),
            TokenInfo(orth = "Ma'am", lemma = "madam", norm = "madam"),
            TokenInfo(orth = "o'clock", lemma = "o'clock", norm = "o'clock"),
            TokenInfo(orth = "O'clock", lemma = "o'clock", norm = "o'clock"),
            TokenInfo(orth = "lovin'", lemma = "love", norm = "loving"),
            TokenInfo(orth = "Lovin'", lemma = "love", norm = "loving"),
            TokenInfo(orth = "lovin", lemma = "love", norm = "loving"),
            TokenInfo(orth = "Lovin", lemma = "love", norm = "loving"),
            TokenInfo(orth = "havin'", lemma = "have", norm = "having"),
            TokenInfo(orth = "Havin'", lemma = "have", norm = "having"),
            TokenInfo(orth = "havin", lemma = "have", norm = "having"),
            TokenInfo(orth = "Havin", lemma = "have", norm = "having"),
            TokenInfo(orth = "doin'", lemma = "do", norm = "doing"),
            TokenInfo(orth = "Doin'", lemma = "do", norm = "doing"),
            TokenInfo(orth = "doin", lemma = "do", norm = "doing"),
            TokenInfo(orth = "Doin", lemma = "do", norm = "doing"),
            TokenInfo(orth = "goin'", lemma = "go", norm = "going"),
            TokenInfo(orth = "Goin'", lemma = "go", norm = "going"),
            TokenInfo(orth = "goin", lemma = "go", norm = "going"),
            TokenInfo(orth = "Goin", lemma = "go", norm = "going"),
            TokenInfo(orth = "Mt.", lemma = "Mount", norm = "Mount"),
            TokenInfo(orth = "Ak.", lemma = "Alaska", norm = "Alaska"),
            TokenInfo(orth = "Ala.", lemma = "Alabama", norm = "Alabama"),
            TokenInfo(orth = "Apr.", lemma = "April", norm = "April"),
            TokenInfo(orth = "Ariz.", lemma = "Arizona", norm = "Arizona"),
            TokenInfo(orth = "Ark.", lemma = "Arkansas", norm = "Arkansas"),
            TokenInfo(orth = "Aug.", lemma = "August", norm = "August"),
            TokenInfo(orth = "Calif.", lemma = "California", norm = "California"),
            TokenInfo(orth = "Colo.", lemma = "Colorado", norm = "Colorado"),
            TokenInfo(orth = "Conn.", lemma = "Connecticut", norm = "Connecticut"),
            TokenInfo(orth = "Dec.", lemma = "December", norm = "December"),
            TokenInfo(orth = "Del.", lemma = "Delaware", norm = "Delaware"),
            TokenInfo(orth = "Feb.", lemma = "February", norm = "February"),
            TokenInfo(orth = "Fla.", lemma = "Florida", norm = "Florida"),
            TokenInfo(orth = "Ga.", lemma = "Georgia", norm = "Georgia"),
            TokenInfo(orth = "Ia.", lemma = "Iowa", norm = "Iowa"),
            TokenInfo(orth = "Id.", lemma = "Idaho", norm = "Idaho"),
            TokenInfo(orth = "Ill.", lemma = "Illinois", norm = "Illinois"),
            TokenInfo(orth = "Ind.", lemma = "Indiana", norm = "Indiana"),
            TokenInfo(orth = "Jan.", lemma = "January", norm = "January"),
            TokenInfo(orth = "Jul.", lemma = "July", norm = "July"),
            TokenInfo(orth = "Jun.", lemma = "June", norm = "June"),
            TokenInfo(orth = "Kan.", lemma = "Kansas", norm = "Kansas"),
            TokenInfo(orth = "Kans.", lemma = "Kansas", norm = "Kansas"),
            TokenInfo(orth =  "Ky.", lemma = "Kentucky", norm = "Kentucky"),
            TokenInfo(orth = "La.", lemma = "Louisiana", norm = "Louisiana"),
            TokenInfo(orth = "Mar.", lemma = "March", norm = "March"),
            TokenInfo(orth = "Mass.", lemma = "Massachusetts", norm = "Massachusetts"),
            TokenInfo(orth = "May.", lemma = "May", norm = "May"),
            TokenInfo(orth = "Mich.", lemma = "Michigan", norm = "Michigan"),
            TokenInfo(orth = "Minn.", lemma = "Minnesota", norm = "Minnesota"),
            TokenInfo(orth = "Miss.", lemma = "Mississippi", norm = "Mississippi"),
            TokenInfo(orth = "N.C.", lemma = "North Carolina", norm = "North Carolina"),
            TokenInfo(orth = "N.D.", lemma = "North Dakota", norm = "North Dakota"),
            TokenInfo(orth = "N.H.", lemma = "New Hampshire", norm = "New Hampshire"),
            TokenInfo(orth = "N.J.", lemma = "New Jersey", norm = "New Jersey"),
            TokenInfo(orth = "N.M.", lemma = "New Mexico", norm = "New Mexico"),
            TokenInfo(orth = "N.Y.", lemma = "New York", norm = "New York"),
            TokenInfo(orth = "Neb.", lemma = "Nebraska", norm = "Nebraska"),
            TokenInfo(orth = "Nebr.", lemma = "Nebraska", norm = "Nebraska"),
            TokenInfo(orth = "Nev.", lemma = "Nevada", norm = "Nevada"),
            TokenInfo(orth = "Nov.", lemma = "November", norm = "November"),
            TokenInfo(orth = "Oct.", lemma = "October", norm = "October"),
            TokenInfo(orth = "Okla.", lemma = "Oklahoma", norm = "Oklahoma"),
            TokenInfo(orth = "Ore.", lemma = "Oregon", norm = "Oregon"),
            TokenInfo(orth = "Pa.", lemma = "Pennsylvania", norm = "Pennsylvania"),
            TokenInfo(orth = "S.C.", lemma = "South Carolina", norm = "South Carolina"),
            TokenInfo(orth = "Sep.", lemma = "September", norm = "September"),
            TokenInfo(orth = "Sept.", lemma = "September", norm = "September"),
            TokenInfo(orth = "Tenn.", lemma = "Tennessee", norm = "Tennessee"),
            TokenInfo(orth = "Va.", lemma = "Virginia", norm = "Virginia"),
            TokenInfo(orth = "Wash.", lemma = "Washington", norm = "Washington"),
            TokenInfo(orth = "Wis.", lemma = "Wisconsin", norm = "Wisconsin"))

        for (abb in abbreviations){
            exceptions[abb.orth] = listOf(abb)
        }

        val anotherOneExceptions = listOf("'d",
            "a.m.",
            "Adm.",
            "Bros.",
            "co.",
            "Co.",
            "Corp.",
            "D.C.",
            "Dr.",
            "e.g.",
            "E.g.",
            "E.G.",
            "Gen.",
            "Gov.",
            "i.e.",
            "I.e.",
            "I.E.",
            "Inc.",
            "Jr.",
            "Ltd.",
            "Md.",
            "Messrs.",
            "Mo.",
            "Mont.",
            "Mr.",
            "Mrs.",
            "Ms.",
            "p.m.",
            "Ph.D.",
            "Prof.",
            "Rep.",
            "Rev.",
            "Sen.",
            "St.",
            "vs.",
            "v.s.")

        for (exception in anotherOneExceptions){
            exceptions[exception] = listOf(TokenInfo(orth = exception))
        }

        for (excludeString in exclude){
            if (excludeString in exceptions.keys){
                exceptions.remove(excludeString)
            }
        }
    }
}
