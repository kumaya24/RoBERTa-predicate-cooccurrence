# Template options
TEMPLATE_OPTIONS = {
    # Nominalization
    "nom_vintran": [
        # "In English, a <mask> is defined as when some entities {w}.",
        # "In English, an <mask> is defined as when some entities {w}.",
        # "In English, some <mask> is defined as when some entities {w}.",
        # "In English, the <mask> is defined as when some entities {w}."
        # "Technically, a <mask> is defined as when some entities {w}.",
        # "Technically, an <mask> is defined as when some entities {w}.",
        # "Technically, some <mask> is defined as when some entities {w}.",
        # "Technically, the <mask> is defined as when some entities {w}."
        # "Literally, a <mask> is defined as when some entities {w}.",
        # "Literally, an <mask> is defined as when some entities {w}.",
        # "Literally, the <mask> is defined as when some entities {w}.",
        # "Literally, some <mask> is defined as when some entities {w}."
        #"A <mask> is exclusively defined as when some entities {w}.",
        #"An <mask> is exclusively defined as when some entities {w}.",
        #"Some <mask> is exclusively defined as when some entities {w}.",
        #"The <mask> is exclusively defined as when some entities {w}."
        # "A <mask> is defined as the state in which some entities {w}.",
        # "An <mask> is defined as the state in which some entities {w}.",
        # "The <mask> is defined as the state in which some entities {w}.",
        # "Some <mask> is defined as the state in which some entities {w}."
        #"A <mask> is defined as a way when someone or something {w}.",
        #"An <mask> is defined as a way when someone or something {w}.",
        #"Some <mask> is defined as a way when someone or something {w}.",
        #"The <mask> is defined as a way when someone or something {w}."
    # NOT BAD: "The action that people {w} is defined as <mask>.",
        # "The meaning of <mask> is for people to {w}.",
        # "<mask> is created for some entities to {w}.",
        # "<mask> is defined as a practice through which people {w}."

        
        #"A <mask> is defined as a way to describe when someone or something {w}.",
        #"An <mask> is defined as a way to describe when someone or something {w}.",
        #"Some <mask> is defined as a way to describe when someone or something {w}.",
        #"The <mask> is defined as a way to describe when someone or something {w}."
        #"A <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}.",
        #"An <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}.",
        #"The <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}.",
        #"Some <mask> is defined as when some entities {w}, or as a way to describe when someone or something {w}."
        
        
        
        # When only use 2nd set of prompts, "technically" makes it worse. 

        # Do -> Deed (good)
        # Tell -> tale (too low)
        # Try -> trial (no)
        # Remember -> remembrance (good)
        # Eat -> event as first place..., eating is first place when only 2nd sets of prompts
        # Starve (bad())

        #"In strict terms, a <mask> is defined as when some entities {w}.",
        #"In strict terms, an <mask> is defined as when some entities {w}.",
        #"In strict terms, the <mask> is defined as when some entities {w}.",
        #"In strict terms, some <mask> is defined as when some entities {w}."
        # X tale, x deed, x trial, x remembrance
        #"{w} is defined as to cause someone to come to a <mask>."

        # -->>>>
        "A <mask> is defined as when some entities {w}.",
        "An <mask> is defined as when some entities {w}.",
        "Some <mask> is defined as when some entities {w}.",
        "The <mask> is defined as when some entities {w}.",
        
        
        "A <mask> is defined as the act of {w}.",
        "An <mask> is defined as the act of {w}.",
        "The <mask> is defined as the act of {w}.",
        "Some <mask> is defined as the act of {w}.",

    ],
    "nom_vtran": [
        "A <mask> is defined as when some entities {w} something.",
        "An <mask> is defined as when some entities {w} something.",
        "The <mask> is defined as when some entities {w} something.",
        "Some <mask> is defined as when some entities {w} something.",
        #"Technically, a <mask> is defined as when some entities {w} something.",
        #"Technically, an <mask> is defined as when some entities {w} something.",
        #"Technically, the <mask> is defined as when some entities {w} something.",
        #"Technically, some <mask> is defined as when some entities {w} something.",
        #"In English, a <mask> is literally defined as when some entities {w} something.",
        #"In English, an <mask> is literally defined as when some entities {w} something.",
        #"In English, the <mask> is literally defined as when some entities {w} something.",
        #"In English, some <mask> is literally defined as when some entities {w} something.",

        # "The thing that people {w} something is defined as <mask>."
        # "The action that people {w} something is defined as <mask>."
        # "<mask> is to describe that people {w} something."

        # Show many og form as well
        "A <mask> is defined as the act of {w} something.",
        "An <mask> is defined as the act of {w} something.",
        "The <mask> is defined as the act of {w} something.",
        "Some <mask> is defined as the act of {w} something.",

       
    ],
    "agent_vintran":[
        #"They are a <mask> because they {w} something.",
        #"They are an <mask> because they {w} something.",
        #"They are the <mask> because they {w} something.",
        # "People who {w} something is identified as a <mask>."
         "Person who do the action to {w} is identified as a <mask>."
        # "The person who performs the action of to {w} is identified as a <mask>."
        #"A <mask> is a person who {w} something.",
        #"An <mask> is a person who {w} something.",
        #"The <mask> is a person who {w} something."
        # "The person is called as a <mask> because the main thing that they do is to {W} something.",
        # "The person is called as an <mask> because the main thing that they do is to {W} something.",
        # "The person is called as the <mask> because the main thing that they do is to {W} something."
        # "The main thing that I do is to {w} something, so I am called as <mask>.",
        # "I am called as a <mask> because the main thing that I do is to {w} something.",
        # "I am called as an <mask> because the main thing that I do is to {w} something.",
        # "I am called as the <mask> because the main thing that I do is to {w} something."

        #"Technically, A <mask> is defined as someone or something that will {w}.",
        #"Technically, An <mask> is defined as someone or something that will {w}.",
        #"Technically, The <mask> is defined as someone or something that will {w}.",
        #"Technically, Some <mask> is defined as someone or something that will {w}."

        # "<mask> is defined as the person who {w}."
        # "<mask> is the profession that {w} regularly.",

        #"In English, a <mask> is defined as someone who will {w} regularly.",
        #"In English, an <mask> is defined as someone who will {w} regularly.",
        #"In English, the <mask> is defined as someone who will {w} regularly.",
        # "In English, some <mask> is defined as someone who will {w} regularly."

        #"A <mask> refers to a person who {w} regularly.",
        #"An <mask> refers to a person who {w} regularly.",
        #"The <mask> refers to a person who {w} regularly.", 
        #"A <mask> refers to a person who {w} frequently.",
        #"An <mask> refers to a person who {w} frequently.",
        #"The <mask> refers to a person who {w} frequently.", 
        #"Some <mask> refers to a person who {w} regularly."
        #"Technically, something or someone able to {w} whom is defined as a <mask>.",
        #"Technically, something or someone able to {w} whom is defined as an <mask>.",
        #"Technically, something or someone able to {w} whom is defined as the <mask>."
        
        #"If someone or something {w} frequently, they are therefore a <mask>.",
        #"If someone or something {w} frequently, they are therefore an <mask>.",
        #"If someone or something {w} frequently, they are therefore the <mask>."
    ],
    "agent_vtran":[
        # "He is called as a <mask> because the main thing that he does is to {w} something.",
        # "He is called as an <mask> because the main thing that he does is to {w} something.",
        # "He is called as the <mask> because the main thing that he does is to {w} something.",
        # "She is called as a <mask> because the main thing that she does is to {w} something.",
        # "She is called as an <mask> because the main thing that she does is to {w} something.",
        # "She is called as the <mask> because the main thing that she does is to {w} something."

        #"A <mask> is defined as someone that will {w} someone or something.",
        #"An <mask> is defined as someone that will {w} someone or something.",
        # "<mask> is defined as someone or something that will {w} something."

        #"Technically, A <mask> is defined as someone who {w} something regularly.",
        #"Technically, An <mask> is defined as someone who {w} something regularly.",
        #"Technically, The <mask> is defined as someone who {w} something regularly.",
        #"Technically, Some <mask> is defined as someone who {w} something regularly."


        #"A <mask> is defined as someone who {w} something regularly.",
        #"An <mask> is defined as someone who {w} something regularly.",     
        #"The <mask> is defined as someone who {w} something regularly.",
        #"Some <mask> is defined as someone who {w} something regularly.",

        #"A <mask> is defined as someone whose job is to {w} something regularly.",
        #"An <mask> is defined as someone whose job is to {w} something regularly.",
        #"The <mask> is defined as someone whose job is to {w} something regularly.",
        #"Some <mask> is defined as someone whose job is to {w} something regularly."

        "A <mask> is defined as a person whose work is to {w} things.",
        "An <mask> is defined as a person whose work is to {w} things.",
        "The <mask> is defined as a person whose work is to {w} things."
    ],
    
    # Eventuality
    "evt_vintran": [
        # "A <mask> is an eventuality that involves {w}.",
        # "An <mask> is an eventuality that involves {w}.",
        # "Some <mask> is an eventuality that involves {w}."
        # "Technically, <mask> is defined as when someone or something will {w}.",
        # "Technically, a <mask> is defined as when someone or something will {w}.",
        # "Technically, an <mask> is defined as when someone or something will {w}."
        "In English, <mask> is defined as when someone or something will {w}.",
        "In English, a <mask> is defined as when someone or something will {w}.",
        "In English, an <mask> is defined as when someone or something will {w}."
    ],
    "evt_vtran": [
        # "A <mask> is an eventuality that involves {w} something.",
        # "An <mask> is an eventuality that involves {w} something.",
        # "Some <mask> is an eventuality that involves {w} something."

        # "<mask> is defined as when someone or something will {w} someone or something."
        # "A <mask> is defined as when someone or something will {w} someone or something.",
        # "An <mask> is defined as when someone or something will {w} someone or something.",
        #"Some <mask> is defined as when someone or something will {w} someone or something."

        "In English, <mask> is defined as when someone or something will {w} someone or something.",
        "In English, a <mask> is defined as when someone or something will {w} someone or something.",
        "In English, an <mask> is defined as when someone or something will {w} someone or something.",
        "In English, some <mask> is defined as when someone or something will {w} someone or something.",
        "In English, the <mask> is defined as when someone or something will {w} someone or something."
        #"A <mask> is defined as a event of {w} something.",
        #"An <mask> is defined as a event of {w} something.",
        #"The <mask> is defined as a event of {w} something.",
        #"Some <mask> is defined as a event of {w} something."
    ],

    #### V -> ADJ
    "participleAdj_vintran":[
        # "people say 'this is <mask>!' when they {w}."
        "In English, something or someone able to {w} is defined as being <mask>."
    ],
    "participleAdj_vtran":[
        #"people say 'this is <mask>!' when they {w} it.",
        # "Something is very <mask> is defined as when something is capable to be {w}."
        #"The word {w} can be converted to <mask> to describe things."
        # "Very <mask> is defined as when something can {w} something."
        # "In English, the sentence that something can {w} something is as same as the sentence that something is very <mask>."
        # "In English, they can {w} something, which also means that something is very <mask>."
        # "In English, something or someone able to {w} something is defined as being <mask>."
        "To {w} someone or something is defined as the causation of them to be <mask> someone or something."
    ],
    
    ##### Causative
    "causative_vintran": [
        #"In English, for someone or something to <mask> someone or something is defined as to cause it to {w}.",
        # "Someone or something to <mask> someone or something causes the someone or something else to {w}."
        # "In English, for someone or something to <mask> someone or something is defined as to cause the latter one to {w}.",
        "In English, for someone or something to <mask> things is defined as to cause them to {w}.",
    ],
    "causative_vtran": [
        #"In English, for someone or something to {w} someone or something is defined as to cause it to <mask>.",
        #"In English, for someone or something to {w} someone or something is defined as to cause the latter one to <mask>.",
        # "In English, someone or something is being {w} because someone or something else must <mask>."
        # "To {w} someone or something causes them to <mask>."
        # "To {w} someone or something is defined as causing them to <mask>."
        #"{w} someone or something causes them to <mask>."
        "In English, for someone or something to {w} things is defined as to cause them to <mask>.",
        
    ],
    ## DISCARDED
        #result
    "res_vintran": [
        # "When people {w}, the item that they create is called as <mask>."
        #"The thing that people {w} is <mask>."
        # "<mask> is created because people {w}."
        # "The <mask> is the output result if people {w}."
        # "The item is called as a <mask> because it is the result or the act to {w} .",
        # "The item is called as an <mask> because it is the result or the act to {w} .",
        # "The item is called as the <mask> because it is the result or the act to {w} ."
        # "Technically, <mask> is defined as the result when someone or something use it to {w}."
        "In English, <mask> is defined as the result when someone or something {w}."
    ],
    "res_vtran": [
        # "When people {w}, the item that they create is called as <mask>."
        #"The thing that people {w} is <mask>."
        # "<mask> is created because people {w}."
        # "The <mask> is the output result if people {w}."
        "The item is called as a <mask> because it is the result or the act to {w} something.",
        "The item is called as an <mask> because it is the result or the act to {w} something.",
        "The item is called as the <mask> because it is the result or the act to {w} something."
    ],
    "item_vintran":[
        # "The item is called as a <mask> because the main thing that it does is to {w} something.",
        # "The item is called as an <mask> because the main thing that it does is to {w} something.",
        # "The item is called as the <mask> because the main thing that it does is to {w} something.",

        #"A <mask> is defined as something that will {w}.",
        #"An <mask> is defined as something that will {w}."

        # "In English, <mask> is defined as something that will {w}."
        "A <mask> is defined as something that {w} regularly.",
        "An <mask> is defined as something that {w} regularly.",
        "The <mask> is defined as something that {w} regularly.",
        "Some <mask> is defined as something that {w} regularly.",
    ],
    "item_vtran":[
        "The item is called as a <mask> because the main thing that it does is to {w} something.",
        "The item is called as an <mask> because the main thing that it does is to {w} something.",
        "The item is called as the <mask> because the main thing that it does is to {w} something .",
    ],
    "state_vintran": [
        "<mask> is the state of to {w}."
    ],
    "state_vtran": [
        "<mask> is the state of to {w} something."
    ],
    
    # Instrument
    
    "inst_vintran": [
        # "The item is called as a <mask> because the function of it is to {w}.",
        # "The item is called as an <mask> because the main thing that it does is to {w}.",
        # "The item is called as the <mask> because the main thing that it does is to {w}."

        "Technically, <mask> is defined as when someone or something use it to {w} ."
    ],
    "inst_vtran": [
        # "The item is called as a <mask> because it is the instrument to {w}.",
        "The item is called as a <mask> because the main thing that it does is to {w} something.",
        "The item is called as an <mask> because the main thing that it does is to {w} something.",
        "The item is called as the <mask> because the main thing that it does is to {w} something."
        # "A <mask> is needed to {w}",
        # "The <mask> is needed to {w}."
    ],

}