# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:25:55 2021

@author: José Carlos Costa
email: carlosjuniorcosta1@gmail.com
"""



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import re
import xml.etree.ElementTree as ET
from collections import Counter
import pickle
import nltk
from nltk import word_tokenize
import numpy as np
import os
from string import punctuation
nltk.download('punkt')
import xml.etree.ElementTree as ET


def limpa_coral(texto):
    import re 
    texto = re.sub(r'(\*\w{3}\:)?(\[\d+\])?|\[?\/\d+\]?|\+|/|/{2}|=?i?\s?\-?\w{3}_?r?s?n?=\s?\$?', '', texto)
    texto = texto.replace('hhh', '').replace('yyyy', '')\
    .replace('yyy', '').replace('xxx', '').replace('<', '')\
    .replace('>','').replace('?', '').replace('=', '')
    
    texto = re.sub(r'&\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto   

def limpa_coral_sobreposicao(texto):
    import re 
    texto = re.sub(r'(\*\w{3}\:)?(\[\d+\])?|\[?\/\d+\]?|\+|/|/{2}|=?i?\s?\-?\w{3}_?r?s?n?=\s?\$?', '', texto)
    texto = texto.replace('hhh', '').replace('yyyy', '').replace('yyy', '').replace('xxx', '')\
    .replace('?', '').replace('=', '')
    
    texto = re.sub(r'&\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto


def normaliza_coral(texto):
    import re 
    texto = re.sub(r'(\*\w{3}\:)?(\[\d+\])?|\[?\/\d+\]?|\+|/|/{2}|=?i?\s?\-?\w{3}_?r?s?n?=\s?\$?', '', texto)
    texto = texto.replace('hhh', '').replace('yyyy', '')\
    .replace('yyy', '').replace('xxx', '').replace('<', '')\
    .replace('>','').replace('?', '').replace('=', '')
    texto = re.sub('&\w+', '', texto)
    texto = re.sub(r'&\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.replace("'", "’")
    texto = texto.strip()
    formas_conv ="""
    ni (em), a’ (olha), acabamo (acabamos), achamo (achamos), agradecemo (agradecemos), a’ lá (olha), a’ (olha), a’ (olha), aprendemo (aprendemos), arrumamo (arrumamos), assinávamo (assinávamos), atravessamo (atravessamos), avi (vi), avinha (vinha), bebemo (bebemos), beijamo (beijamos), botemo (botamos), chegamo (chegamos), cheguemo (chegamos), choramo (choramos), colocamo (colocamos), começamo (começamos), comemo (comemos), comemoramo (comemoramos), compramo (compramos), conhecemo (conhecemos), conseguimo (conseguimos), contamo (contamos), conversamo (conversamos), corremo (corremos), cortamo (cortamos), deixamo (deixamos), descansamo (descansamos), descemo (descemos), devemo (devemos), empurramo (empurramos), encontramo (encontramos), entramo (entramos), envem (vem), envinha (vinha), escolhemo (escolhemos), esquecemo (esquecemos), estamo (estamos), estudemo (estudamos), evem (vem), falamo (falamos), fazido (feito), ficamo (ficamos), fize (fiz), fizemo (fizemos), fomo (fomos), for (formos), fraga (flagra), fragando (flagrando), frago (flagro), fumo (fomos), ganhamo (ganhamos), levamo (levamos), levantamo (levantamos), levantemo (levantamos), mandamo (mandamos), manti (mantive), o’ (olha), o’(olha), paramo (paramos), passamo (passamos), pedimo (pedimos), peguemo (pegamos), perdemo (perdemos), pinchando (pichando), pintemo (pintemos), podemo (podemos), precisamo (precisamos), pusemo (pusemos), resolvemo (resolvemos), saímo (saímos), seje (seja), sentamo (sentamos), sentemo (sentamos), separamo (separamos), somo (somos), sufro (sofro), temo (temos), tiramo (tiramos), tivemo (tivemos), tomamo (tomamos), trabalhamo (trabalhamos), trago (trazido), vesse (visse), viemo (viemos), vimo (vimos), tó (toma), cê (você), cês (vocês), e’ (ele), ea (ela), eas (elas), es (eles), ocê (você), ocês (vocês), aque’ (aquele), aquea (aquela), aqueas (aquelas), aques (aqueles), ca (com a), co (com o), cos (com os), cum (com um), cuma (com uma), d’ (de) d’(de), d’(de), d’(de), dum (de um), duma (de uma), dumas (de umas), duns (de uns),  deerreí (na DRI), ni (em), num (em um), numa (em uma), numas (em umas), pa (para), pas (para as), p’(para), p’(para), p’(para), p’(para), p’(para), p’(para), p’(para), po (para o), p’(para), pos (para os), p’(para), pra (para), pr’(para), pras (para as), pro (para o), pr’(para), pros (para os), prum (para um), pruma (para uma), pruns (para uns), p’(para), p’ (para), p’(para), pum (para um), puma (para uma), c’ aqueas (com aquelas), c’(com), c’ cê (com você), c’ e’ (com ele), c’ (com), c’(com essas), c’(com), c’ ocê (com você), c’ ocês (com vocês), daque’ (daquele), daquea (daquela), daqueas (daquelas), daques (daqueles), d’ cê (de você), de’ (dele), dea (dela), d’(de), d’(de), d’(de), des (deles), d’ es (de eles), d’(de), d’ ocê (de você), d’ ocês (de vocês), naque’ (naquele), naquea (naquela), naques (naqueles), ne’ (nele), n’ ocê (em você), n’ ocês (em vocês), p’ aque’ (para aquele), p’(para), p’ cê (para você), p’ cês (para vocês), p’ e’ (para ele), p’(para), p’ (para), p’(para), p’ es (para eles), p’ esse (para esse), p’ mim (para mim), p’ ocê (para você), p’ ocês (para vocês), pr’(para), pr’(para), pr’(para), pr’(para), pr’ ocê (para você), pr’ ocês (para vocês), p’ sio’ (para a senhora), p’ siora (para a senhora), p’(para), armoçar (almoçar), artinho (altinho), arto (alto), arto (alto), comprica (complica), compricar (complicar), cravícula (clavícula), escardada (escaldada), prano (plano), pranta (planta), pray (play), prissado (plissado), probremas (problemas), sortando (soltando), sortar (soltar), sortei (soltei), sorto (solto), sortou (soltou), vorta (volta), vortar (voltar), vortava (voltava), vorto (volto), nũ (não), canarim (canarinho), espim (espinho), padrim (padrinho), passarim (passarinho), porco-espim (porco-espinho), sozim (sozinho), almoçozim (almoçozinho), amarelim (amarelinho), azulzim (azulzinho), bebezim (bebezinho), bichim (bichinho), bocadim (bocadinho), bonitim (bonitinho), cachorrim (cachorrinho), cantim (cantinho), capoeirim (capoeirinhas), carrim (carrinho), cedezinho (CD), certim (certinho), certins (certinhos), Chapeuzim Vermelho (Chapeuzinho Vermelho), chazim (chazinho), controladim (controladinha), desfiadim (desfiadinho), direitim (direitinho), direitim (direitinho), esquisitim (esquisitinho), fechadim (fechadinho), filhotim (filhotinho), formulariozim (formulariozinho), fundim (fundinho), Geraldim (Geraldinho), golezim (golezinho), igualzim (igualzinho), instantim (instantinho), jeitim (jeitinho), Joãozim (Joãozinho), joguim (joguinho), ladim (ladinho), maciim (maciinho), mansim (mansinho), Marquim (Marquinho), meninim (menininho), morenim (moreninho), murim (murinho), Paulim (Paulinho), pequeninim (pequenininha), pertim (pertinho), negocim (negocinhos), partidim (partidinho), porquim (porquinho), portim (portinha), potim (potinho), pouquim (pouquinho), pozim (pozinho), pretim (pretinho), prontim (prontinho), quadradim (quadradinha), quadradim (quadradinho), queimadim (queimadinho), rapidim (rapidinho), recheadim (recheadinho), rolim (rolinho), tamanim (tamaninho), tampadim (tampadinho), terrenim (terreninho), tiquim (tiquinho), todim (todinho), toquim (toquinho), trancadim (trancadinhos), trenzim (trenzinho), tudim (tudinho), sio’ (senhora), sior (senhor), siora (senhora), sô (senhor), mó (muito), po’ (pode) ,tá (está) ,tamo (estamos) ,tamos (estamos) ,tão (estão) ,tar (estar) ,taria (estaria) ,tás (estás) ,tava (estava) ,tavam (estavam) ,távamos (estávamos) ,tavas (estavas) ,teja (esteja),teve (esteve) ,tive (estive) ,tiver (estiver) ,tiverem (estiverem) ,tivesse (estivesse) ,tô (estou) ,vamo (vamos) ,vão (vamos) ,vim (vir) ,xá (deixa), antiguim (antiguinho), banhozim (banhozinho), branquim (branquinho), certim (certinho), 
    devagarzim (devagarzinho), direitim (direitinho), direitim (direitinho), gostosim (gostosinho), limãozim (limãozinho), minutim (minutinho), pertim (pertinho), pulim (pulinho), pouquim (pouquinho), rapidim (rapidinho), recibim (recibinho), verdim (verdinho), xixizim (xixizinho), zerim (zerinho)
    babacar (embabacar), zucrinando (azucrinando), zucrinar (azucrinar), brigado (obrigado), brigada (obrigada), baixa (abaixa), credita (acredita), creditei (acreditei), creditou (acreditou) , baixar (abaixar), baixei (abaixei), baulado (abaulado), bora (embora), borrecido (aborrecido), brigada (obrigada), brigado (obrigado), caba (acaba), cabar (acabar), cabava (acabava), cabei (acabei), cabou (acabou), celera (acelera), celerando (acelerando), certar (acertar), chei (achei), cho (acho), contece (acontece), contecer (acontecer), conteceu (aconteceu), cordava (acordava), creditei (acreditei), dianta (adianta), doro (adoro), dotada (adotada), fessora (professora), final (afinal), fundar (afundar), garrado (agarrados), garrou (agarrou), gora (agora), gual (igual), gualzim (igualzinho), guenta (aguenta), guentando (aguentando), guentar (aguentar), guento (aguento), guentou (aguentou), inda (ainda), judar (ajudar), lambique (alambique), laranjado (alaranjado), lisou (alisou), magina (imagina), mamentar (amamentar), manhã (amanhã), marelo (amarelo), marrava (amarrava), migão (amigão aumentativo), mor (amor), ném (neném), panhava (apanhava), parece (aparece), pareceu (apareceu), partamento (apartamento), pelido (apelido),  perta (aperta), pertar (apertar), pertei (apertei), pesar (apesar), pinhada (apinhada), proveita (aproveita), proveitei (aproveitei), proveitou (aproveitou),  proveitando (aproveitando), proveitei (aproveitei), purra (empurra), qui (daqui), rancaram (arrancaram), rancava (arrancava), rancou (arrancou), ranjar (arranjar), ranjasse (arranjasse), ranjou (arranjou), rebentando (arrebentando), rebentar (arrebentar), regaço (arregaços), rorosa (horrorosa), roz (arroz), rumaram (arrumaram), sobiando (assobiando),té (até), té (até), teirinho (inteirinho), teja (esteja), tendeu (entendeu), tendi (entendi), testino (intestino), tradinha (entradinha), trapalha (atrapalha), trapalhado (atrapalhado), trapalhou (atrapalhou), travessa (atravessa), travessadinho (atravessadinho), trevida (atrevida), trevido (atrevido), trevidão (atrevidão), vó (avó), vô (avô)""" 
   

    regex_apostr = r"(\w+’(?!\n))\s(\w+(?!\n))"
    regex_excep= r"([A-Za-z]+’)([A-Za-z]+)"
    
    texto = texto.replace("'", "’")
    texto = re.sub(regex_apostr, r"\1\2", texto)
    
    formas_conv = re.sub(regex_apostr, r"\1\2", formas_conv)
    
    tuplas = re.findall(r"(\w+|\w+\s\w+|\w+’|\w+’\s?\w+|\w+\s\w+\s\w+’|\w+\s?\w+’)\s?\((\w+|\w+\s\w+)\)", formas_conv)
    tuplas=  [(x[0].strip(), x[1]) for x in tuplas]

    dicio = dict(tuplas)
        
    texto = " ".join([dicio[p] if p in dicio else p for p in texto.split(' ')])
        
    texto= re.sub(regex_excep, r'\1 \2', texto)
    
    formas_conv = re.sub(regex_excep, r"\1 \2", formas_conv)
    
    tuplas = re.findall(r"(\w+|\w+\s\w+|\w+’|\w+’\s?\w+|\w+\s\w+\s\w+’|\w+\s?\w+’)\s?\((\w+|\w+\s\w+)\)", formas_conv)
    
    dicio = dict(tuplas)
    
    texto = texto.replace('\n', '\n$ ')
    
    texto =  '\n'.join([dicio[x] if x in dicio else x for x in texto.split(' ')])
        
    texto = texto.replace('\n', ' ')
        
    texto = texto.split('$')
    texto = '\n'.join([x.strip() for x in texto])
    
    texto = texto.replace('o’', 'olha').replace('pa’', 'para')\
        .replace('Vix’', 'Vixe').replace('No’', 'Nossa').replace('pr’', 'para')\
            .replace('n’', 'não').replace('e’', 'ele').replace('Nu’', 'Nossa')
    texto = re.sub(r'i?-?_?COB_?s?r?|i?-?_?COM_?s?r?|i?-?_?APC_?s?r?|i?-?_?CMM_?s?r?|i?-?_?TOP_?s?r?=|i?-?_?TPL_?s?r?|i?-?_?APT_?s?r?|i?-?_?PAR_?s?r?=|i?-?_?PAR_?s?r?|i?-?_?INT_?s?r?|i?-?_?SCA_?s?r?|i?-?_?AUX_?s?r?|i?-?_?PHA_?s?r?|i?-?_?ALL_?s?r?|i?-?_?CNT_?s?r?|i?-?_?DCT_?s?r?|i?-?_?EXP_?s?r?|i?-?_?DCT_?s?r?', '', texto)
    texto = texto.strip()
    
    
    return texto



file_1 = '\n'.join([x for x in os.listdir() if x.endswith('xml')])
utterances = []
participant = []
start_time = []
end_time = []
   
audio = []


for filename in file_1.split():
    with open(filename, 'r', encoding="utf-8") as content:
        tree = ET.parse(content)
        root = tree.getroot()
        for y in root.iter('UNIT'):                
            utterances.append(y.text.strip())
        
            participant.append(y.get('speaker'))
            start_time.append((y.get('startTime')))
            end_time.append((y.get('endTime')))
            audio.append(filename)
            start_time_f = re.findall(r'\d+.\d+', '\n'.join(start_time))
            end_time_f = re.findall(r'\d+.\d+', '\n'.join(end_time))


df = pd.DataFrame()
df['utterances'] = utterances
df['participant'] = participant
df['audio'] = audio
             
df['COB'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?COB_?s?r?=', x)))
df['COM'] =  df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?COM_?s?r?=', x)))
df['APC'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?APC_?s?r?=', x)))
df['CMM'] =  df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?CMM_?s?r?=', x)))
df['TOP_TPL'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?TOP_?s?r?=|=i?-?_?TPL_?s?r?=', x)))
df['APT'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?APT_?s?r?=', x)))
df['PAR_PRL'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?PAR_?s?r?=|=i?-?_?PAR_?s?r?=', x)))
df['INT'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?INT_?s?r?=', x)))
df['SCA'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?SCA_?s?r?=', x)))
df['textual_units'] = df['COM'] + df['APC'] + df['CMM'] + df['TOP_TPL'] + df['APT'] + df['PAR_PRL'] + df['INT']
df['tonal_units'] = df['utterances'].apply(lambda x: len(re.findall(r'(?<!\n)/(?!\d)', x)))
df['AUX'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?AUX_?s?r?=', x)))
df['PHA'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?PHA_?s?r?=', x)))
df['ALL'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?ALL_?s?r?=', x)))
df['CNT'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?CNT_?s?r?=', x)))
df['DCT'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?DCT_?s?r?=', x)))
df['EXP'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?EXP_?s?r?=', x)))
df['INP'] = df['utterances'].apply(lambda x: len(re.findall(r'=i?-?_?DCT_?s?r?=', x)))
df['inter_utterance'] = df['utterances'].apply(lambda x: len(re.findall(r'\+', x)))
df['start_time'] = start_time_f
df['end_time'] = end_time_f
df[['start_time', 'end_time']] = df[['start_time', 'end_time']].astype('float')
df['ut_length'] = round(df['end_time'] - df['start_time'], 3)
df['cleaned_utterances'] = df['utterances'].apply(limpa_coral)
df['number_of_words'] = df['cleaned_utterances'].apply(lambda x: len(x.split()))
df['words_per_second'] = (df['number_of_words'] / df['ut_length']).round(3)
df['corpus'] = df['audio'].apply(lambda x: 'Coral_brasil' if x.startswith('b') else 'Coral_esq')
# df['cleaned_for_overlapping'] = df_doc['utterances'].apply(limpa_coral_sobreposicao)
# df['overlapping_words'] = df['cleaned_for_overlapping'].apply(lambda x: ' '.join(re.findall(r'<.+>', str(x))))
# df['cleaned_overlapping'] = df['overlapping_words'].apply(limpa_coral)
# df['number_overlapping'] = df['cleaned_overlapping'].apply(lambda x: len(x.split()))  
# df.drop(['cleaned_for_overlapping', 'cleaned_overlapping'], axis = 1, inplace = True)
df['retractings'] = df['utterances'].apply(lambda x: len(re.findall(r'/(=?\d+)', x)))
df['retr_words'] = df['utterances'].apply(lambda x: sum(map(int, re.findall(r'\d+', x))))
df['TMT'] = df['utterances'].apply(lambda x: len(re.findall(r'.?&he', x)))
df['all_patterns'] = df['utterances'].apply(lambda x: ' '.join(re.findall(r'=i?_?-?\w{3}_?r?s?=', x)))
df['filtered_patterns'] = df['all_patterns'].apply(lambda x: re.sub(r'(=SCA=|=EMP=|=UNC=|=TMT=)', '', x))
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'\s+', ' ', x)).str.strip()
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'(?<=i)_?-?\s?(?=\w+)', '-', x))
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'\s+', ' ', x)).str.strip()
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'_r(?==)|_r(?=\s)', '', x))
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'\s+', ' ', x)).str.strip()
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'_s(?==)|_s(?=\s)', '', x))
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'\s+', ' ', x)).str.strip()
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'i-|i\s(?=\w+)', '', x))
df['filtered_patterns'] = df['filtered_patterns'].apply(lambda x: re.sub(r'\s+', ' ', x)).str.strip()


df['normalized_utterances'] = df['utterances'].apply(normaliza_coral)
    

# df['participant'].unique()


# df.query('participant in "Mailton" or participant in "*DFL:"  or participant \
#          in "Aloysio" or participant in "Regina" or participant in "*CAR:" or \
#              participant in  "Jorge" or audio in "med"')
            
# a = df.query('audio in "med_008_revisado.xml"')

with open('brill_00', 'rb') as f:
    tagueador = pickle.load(f)
    
#depuração normalização    
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"o’\s?|^o’\s?|\s?o’$", 'olha', str(x), flags = re.IGNORECASE))
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"\se’\s?|^e’\s?|\s?e’$", 'ele', str(x), flags = re.IGNORECASE))
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"\sa’\s?|^a’\s?|\s?a’$", 'olha', str(x), flags = re.IGNORECASE))
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"[A-Z]{3}-r|i-[A-Z]{3}", '', str(x), flags = re.IGNORECASE))


print('Tagging with Brill tagger - Mac-Morpho')
df['utterances_POS'] = df['normalized_utterances'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: tagueador.tag(x))

#depuração POS
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='sio’',\s')\w+|(?<='sio',\s')\w+|(?<='senhora',\s')\w+",'PROPESS', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='melancia’',\s')\w+",'N', str(x), flags = re.IGNORECASE))

df['utterances_POS'] = df['utterances_POS'].apply(lambda x:  re.sub(r"\(\'Nossa',\s\'\w+\'\),\s\(\'Senhora\',\s\'\w+\'\),", "('Nossa Senhora', 'IN'),", x))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='na',\s')\w+|(?<='no',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='nessa',\s')\w+|(?<='nesse',\s')\w+|(?<='nisso',\s')\w+|(?<='nesses',\s')\w+|(?<='nessas',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ao',\s')\w+|(?<='aos',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='o',\s')\w+|(?<='os',\s')\w+", 'ART', str(x), flags = re.IGNORECASE))

df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='da',\s')\w+|(?<='do',\s')\w+|(?<='dos',\s')\w+|(?<='das',\s')\w+|(?<='duma',\s')\w+|(?<='dum',\s')\w+|(?<='duns',\s')\w+|(?<='dumas',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='de',\s')\w+", 'PREP', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='pelos',\s')\w+|(?<='pelo',\s')\w+|(?<='pela',\s')\w+|(?<='pelas',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='dele',\s')\w+|(?<='deles',\s')\w+|(?<='dela',\s)\w+|(?<='delas',\s)\w+", 'PROADJ', str(x) , flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='num',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='um',\s')\w+|(?<='uns',\s')\w+|(?<='uma',\s')\w+|(?<='umas',\s')\w+", 'ART', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='comigo',\s')\w+", 'PROPESS', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='por',\s')\w+", 'PREP', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='contigo',\s')\w+", 'PROPESS', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ahn',\s')\w+|(?<='ham',\s')\w+|(?<='hum',\s')\w+|(?<='uhn',\s')\w+", 'IN', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ah',\s')\w+|(?<='eh',\s')\w+|(?<='ih',\s')\w+|(?<='oh',\s')\w+|(?<='ô',\s')\w+|(?<='uai',\s')\w+|(?<='ué',\s')\w+", 'IN', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='nu',\s')\w+|(?<='pá',\s')\w+|(?<='parará',\s')\w+|(?<='tanãnãnã',\s')\w+|(?<='tchan',\s')\w+|(?<='tum',\s')\w+", 'IN', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='deixar',\s')\w+|(?<='deixa',\s')\w+|(?<='deixou',\s')\w+|(?<='deixei',\s')\w+|(?<='deixaram',\s')\w+|(?<='deixamos',\s')\w+|(?<='deixaria',\s')\w+|(?<='deixariam',\s')\w+|(?<='deixam',\s')\w+|(?<='deixo',\s')\w+|(?<='deixasse',\s')\w+|(?<='deixassem',\s')\w+|(?<='deixarmos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='começar',\s')\w+|(?<='começa',\s')\w+|(?<='começou',\s')\w+|(?<='comecei',\s')\w+|(?<='começaram',\s')\w+|(?<='começam',\s')\w+|(?<='começaria',\s')\w+|(?<='começariam',\s')\w+|(?<='começamos',\s')\w+|(?<='começo',\s')\w+|(?<='começasse',\s')\w+|(?<='começassem',\s')\w+|(?<='começarmos',\s')\w+|(?<='comece',\s')\w+|(?<='comecemos',\s')\w+", 'V', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='chegar',\s')\w+|(?<='chega',\s')\w+|(?<='chegou',\s')\w+|(?<='cheguei',\s')\w+|(?<='chegaram',\s')\w+|(?<='chegam',\s')\w+|(?<='chegaria',\s')\w+|(?<='chegariam',\s')\w+|(?<='chegamos',\s')\w+|(?<='chego',\s')\w+|(?<='chegasse',\s')\w+|(?<='chegassem',\s')\w+|(?<='chegarmos',\s')\w+|(?<='chegue',\s')\w+|(?<='cheguemos',\s')\w+", 'V', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ser',\s')\w+|(?<='é',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='somos',\s')\w+|(?<='seria',\s')\w+|(?<='seriam',\s')\w+|(?<='são',\s')\w+|(?<='sou',\s')\w+|(?<='era',\s')\w+|(?<='eram',\s')\w+|(?<='for',\s')\w+|(?<='formos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='sermos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='poder',\s')\w+|(?<='pode',\s')\w+|(?<='pôde',\s')\w+|(?<='pude',\s')\w+|(?<='puderam',\s')\w+|(?<='podemos',\s')\w+|(?<='poderia',\s')\w+|(?<='poderiam',\s')\w+|(?<='podem',\s')\w+|(?<='posso',\s')\w+|(?<='podia',\s')\w+|(?<='podiam',\s')\w+|(?<='pudesse',\s')\w+|(?<='pudessem',\s')\w+|(?<='podermos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='estar',\s')\w+|(?<='está',\s')\w+|(?<='esteve',\s')\w+|(?<='estive',\s')\w+|(?<='estiveram',\s')\w+|(?<='estamos',\s')\w+|(?<='estaria',\s')\w+|(?<='estariam',\s')\w+|(?<='estão',\s')\w+|(?<='estou',\s')\w+|(?<='estava',\s')\w+|(?<='estivemos',\s')\w+|(?<='estivesse',\s')\w+|(?<='estivéssemos',\s')\w+|(?<='estivessem',\s')\w+|(?<='estarmos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ir',\s')\w+|(?<='vai',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='vamos',\s')\w+|(?<='iria',\s')\w+|(?<='iriam',\s')\w+|(?<='vão',\s')\w+|(?<='vou',\s')\w+|(?<='ía',\s')\w+|(?<='fomos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='iria',\s')\w+|(?<='vamos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='nũ',\s')\w+|(?<='né',\s')\w+|(?<='aí',\s')\w+", 'ADV', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='Whatsapp',\s')\w+|(?<='Instagram',\s')\w+|(?<='Facebook',\s')\w+|(?<='big',\s')\w+|(?<='brother',\s')\w+|(?<='buffet',\s')\w+|(?<='feedback',\s')\w+|(?<='fair',\s')\w+|(?<='play',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='open',\s')\w+|(?<='over',\s')\w+|(?<='photoshop',\s')\w+|(?<='pop',\s')\w+|(?<='plus',\s')\w+|(?<='réveillon',\s')\w+|(?<='sexy',\s')\w+|(?<='serial',\s')\w+|(?<='killer',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='shopping',\s')\w+|(?<='short',\s')\w+|(?<='show',\s')\w+|(?<='smartphone',\s')\w+|(?<='software',\s')\w+|(?<='telemarketing',\s')\w+|(?<='videogame',\s')\w+|(?<='tablet',\s')\w+|(?<='Windows',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='yes',\s')\w+|(?<='vip',\s')\w+|(?<='web',\s')\w+|(?<='smartphone',\s')\w+|(?<='slide',\s')\w+|(?<='states',\s')\w+|(?<='videogame',\s')\w+|(?<='online',\s')\w+|(?<='office',\s')\w+|(?<='offline',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ficar',\s')\w+|(?<='fica',\s')\w+|(?<='ficou',\s')\w+|(?<='fiquei',\s')\w+|(?<='ficaram',\s')\w+|(?<='ficamos',\s')\w+|(?<='ficaria',\s')\w+|(?<='ficariam',\s')\w+|(?<='ficam',\s')\w+|(?<='fico',\s')\w+|(?<='ficasse',\s')\w+|(?<='ficassem',\s')\w+|(?<='ficarmos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='né',\s')\w+", 'ADV', str(x), flags = re.IGNORECASE))

#soma etiquetas dialógicas em uma única coluna - mas deixa as que já existem lá 

df_correc_cesq = df.query('corpus in "Coral_esq"')
df_correc_cesq_1 = df_correc_cesq.query('audio in "med_007.xml" or audio in "med_019.xml" or audio in "med_020.xml"')
df_correc_cesq_1['AUX'] = df_correc_cesq_1['PHA'] + df_correc_cesq_1['ALL'] + df_correc_cesq_1['CNT'] + df_correc_cesq_1['DCT'] + df_correc_cesq_1['EXP'] + df_correc_cesq_1['EXP'] + df_correc_cesq_1['INP']
df_correc_cesq_2 = df_correc_cesq.query('audio in "med_008.xml" or audio in "med_013.xml" or audio in "med_015.xml"')
df_correc_cesq = pd.concat([df_correc_cesq_1, df_correc_cesq_2])
df_correc_cb = df.query('corpus in "Coral_brasil"')
df_correc_cb['AUX'] = df_correc_cb['PHA'] + df_correc_cb['ALL'] + df_correc_cb['CNT'] + df_correc_cb['DCT'] + df_correc_cb['EXP'] + df_correc_cb['EXP'] + df_correc_cb['INP']



df = pd.concat([df_correc_cb, df_correc_cesq], ignore_index= True)

#seleciona apenas os pacientes do C-ORAL-ESQ (utilizados com COSTA, 2022)
df = df.query('participant in "MIR" or participant in "CLA" or participant in "GLE" \
         or participant in "DAN" or participant in "MAA" or participant in "VIT" or \
        participant in "Mailton" or participant in "*DFL:" or participant in "Aloysio" \
        or participant in "Regina" or participant in "*CAR:" or participant in "Jorge"')


df['phrases_before_retractings'] = df['utterances'].apply(lambda x: ' '.join(re.findall(r".\w+\s.\w+\s(?=\[/\d\])", x)))
df['phrases_after_retractings'] = df['utterances'].apply(lambda x: ' '.join(re.findall(r"(?<=\[/\d\]\s)\w+\s\w+", x)))

df['dist_retractings'] = df['utterances'].apply(lambda x: ' '.join(re.findall(r'.\w+(?=\s\[\/\d\])|.\w+(?=\s/\d)', x)))


if "Coral_brasil" and "Coral_esq" in df.corpus.values:
    df_cb = df.query('corpus in "Coral_brasil"')

    dist_ret_cb = df_cb['dist_retractings'].tolist()
    list_participants_cb = df_cb['participant'].to_list()
  
    list_ret_cb= []
    
    for x in dist_ret_cb:
        for y in x.split():
            list_ret_cb.append(y)
   
    list_ret_cb = '\n'.join(list_ret_cb)
    list_ret_cb = re.sub('\[|\]|\"|,', '', list_ret_cb)
    
    
    list_ret_cb = [x.lower() for x in list_ret_cb.splitlines() if len(x) > 0]
    
    list_ret_df_cb = pd.DataFrame([x.strip() for x in list_ret_cb], columns=['retractings_full'])
    list_ret_df_cb['corpus'] = 'Coral_brasil'
    
    #cesq
    
    df_cesq = df.query('corpus in "Coral_esq"')

    dist_ret_cesq = df_cesq['dist_retractings'].tolist()
    list_participants_cesq = df_cesq['participant'].to_list()
  
    list_ret_cesq= []
    
    for x in dist_ret_cesq:
        for y in x.split():
            list_ret_cesq.append(y)
   
    list_ret_cesq = '\n'.join(list_ret_cesq)
    list_ret_cesq = re.sub('\[|\]|\"|,', '', list_ret_cesq)
   
    list_ret_cesq = [x.lower() for x in list_ret_cesq.splitlines() if len(x) > 0]
    
    list_ret_df_cesq = pd.DataFrame([x.strip() for x in list_ret_cesq], columns=['retractings_full'])
    list_ret_df_cesq['corpus'] = 'Coral_esq'
    
    list_ret_df = pd.concat([list_ret_df_cb, list_ret_df_cesq])
    
else:
  
    dist_ret = df['dist_retractings'].tolist()
    list_participants = df['participant'].to_list()
  
    list_ret= []
    
    for x in dist_ret:
        for y in x.split():
            list_ret.append(y)
   
    list_ret = '\n'.join(list_ret)
    list_ret = re.sub('\[|\]|\"|,', '', list_ret)
    
    
    list_ret = [x.lower() for x in list_ret.splitlines() if len(x) > 0]
    
    list_ret_df = pd.DataFrame([x.strip() for x in list_ret], columns=['retractings_full'])
    
list_ret_df['retractings'] = list_ret_df['retractings_full'].apply(lambda x: re.sub(r"&|-|<|>|\=", '', x))
list_ret_df['retractings_syl'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r".?uai.?|.?uão.?|.?ai.?|.?ói.?|.?ua.?|.?uo.?|.?io.?|.?ió.?|.?éi.?|.?ei.?|.?ie.?|.?ói.?|.?oi.?|.?au.?|.?ou.?|.?éu.?|.?ui.?|.?a.?|.?á.?|.?â.?|.?ã.?|.?é.?|.?ê.?|.?e.?|.?o.?|.?ô.?|.?õ.?|.?ó.?|.?ò.?|.?i.?|.?í.?",  str(x), flags = re.IGNORECASE)))

list_ret_df['oclus_des'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r"^p|^k|^k$|^c$|^ca|^co|^cu|^cã|^cô|^câ|^cõ|^cú|^q.?.?|^cr|^te(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|u|ú|ó|o|ô|õ)|^té|^tê|^ta|^tá|^tã|^tâ|^to|^tô|^tu|^tú|^tr", x)))
list_ret_df['oclus_voz'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r"^b|^ga|^go|^gu|^gão|^gá|^gó|^gú|^gâ|^gr|^g$|^da|^dá|^dã|^dõ|^dô|^dú|^du|^do|^dr|^de(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)", x)))
list_ret_df['laterais'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r"(^l|^lh)", x)))
list_ret_df['fricativas_des'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r"(^s|^x|^r|^ch|^f|^ce|^ci|^cê|^cí|^cé)", x)))
list_ret_df['fricativas_voz'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r'(^z|^j|^v|^ge|^gi)', x)))
list_ret_df['oclus_nas'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r'(^m|^n)', x)))
list_ret_df['africadas_des'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r'^te(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|a|á|ã|à|â|e|é|ê|i|í|o|ó|õ|ô|u|ú)|^ti|^t$', x)))
list_ret_df['africadas_voz'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r'^de(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)|^di|^d$', x)))
list_ret_df['vogais'] = list_ret_df['retractings'].apply(lambda x: ' '.join(re.findall(r"^a|^e|^i|^o|^u|^ã|^õ|^à|^á|^â|^ô|^ó|^é|^ê|^ú|^h", x)))

list_ret_df = list_ret_df.loc[list_ret_df['retractings_full'] != 'xxx']
list_ret_df = list_ret_df.loc[list_ret_df['retractings_full'] != 'xxxx']
list_ret_df = list_ret_df.loc[list_ret_df['retractings_full'] != 'yyy']
list_ret_df = list_ret_df.loc[list_ret_df['retractings_full'] != 'yyyy']
list_ret_df = list_ret_df.loc[list_ret_df['retractings_full'] != 'hhh']

list_ret_df['retractings'] = list_ret_df['retractings'].apply(lambda x: x[:3] if len(x) > 3 else x)

list_ret_df['oclus_des'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r"^p|^k|^k$|^c$|^ca|^co|^cu|^cã|^cô|^câ|^cõ|^cú|^q.?.?|^cr|^te(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|u|ú|ó|o|ô|õ)|^té|^tê|^ta|^tá|^tã|^tâ|^to|^tô|^tu|^tú|^tr", str(x))))
list_ret_df['oclus_voz'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r"^b|^ga|^go|^gu|^gão|^gá|^gó|^gú|^gâ|^gr|^g$|^da|^dá|^dã|^dõ|^dô|^dú|^du|^do|^dr|^de(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)", x)))
list_ret_df['laterais'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r"(^l|^lh)", x)))
list_ret_df['fricativas_des'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r"(^s|^x|^r|^ch|^f|^ce|^ci|^cê|^cí|^cé)", x)))
list_ret_df['fricativas_voz'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r'(^z|^j|^v|^ge|^gi)', x)))
list_ret_df['oclus_nas'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r'(^m|^n)', x)))
list_ret_df['africadas_des'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r'^te(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|a|á|ã|à|â|e|é|ê|i|í|o|ó|õ|ô|u|ú)|^ti|^t$', x)))
list_ret_df['africadas_voz'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r'^de(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)|^di|^d$', x)))
list_ret_df['vogais'] = list_ret_df['retractings'].apply(lambda x: len(re.findall(r"^a|^e|^i|^o|^u|^ã|^õ|^à|^á|^â|^ô|^ó|^é|^ê|^ú|^h", x)))


if "Coral_brasil" and "Coral_esq" in df.corpus.values:
    retracted_words = pd.DataFrame(list_ret_df.groupby('corpus')['retractings_full'].value_counts())
    retracted_words.columns = ['Frequência']
    retracted_words.reset_index(inplace=True)
    retracted_words.columns = ['corpus', 'retractings', 'Frequência'] 
else:   
    
    contagem_retratadas = pd.DataFrame(list_ret_df['retractings_full'].value_counts())
    contagem_retratadas.reset_index(inplace=True)
    contagem_retratadas.columns = ['retractings', 'Frequência']  

if "Coral_brasil" and "Coral_esq" in df.corpus.values:
    plt.figure(figsize =(9, 5), dpi = 200)
    a = sns.barplot(data = retracted_words.sort_values(by = 'Frequência', ascending =False)[:30], x = 'retractings', y = 'Frequência', hue = 'corpus')
    a.set_title('Palavras retratadas nos corpora', fontsize = 18)
    a.set_ylabel('Quantidade', fontsize = 15)
    a.set_xlabel('Classe do segmento', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper right", frameon=True, fontsize=13)
    
else:
    plt.figure(figsize =(9, 5), dpi = 200)
    a = sns.barplot(data = contagem_retratadas[:30], x = 'retractings', y = 'Frequência')
    a.set_title('Frequência de palavras retratadas', fontsize = 18)
    a.set_ylabel('Quantidade', fontsize = 15)
    a.set_xlabel('Classe do segmento', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper right", frameon=True, fontsize=13)
    
if "Coral_brasil" and "Coral_esq" in df.corpus.values:
    list_ret_df_visu_cb = list_ret_df.query('corpus in "Coral_brasil"')
    list_ret_df_visu_cb.drop(list_ret_df_visu_cb.loc[:, 'retractings_full': 'retractings_syl'], axis = 1, inplace=True)
    # list_ret_df_visu = list_ret_df_visu_cb.loc[:, 'oclus_des':]
    # list_ret_df_visu_cb.drop('corpus', axis =1, inplace= True)
    list_ret_df_visu_cb = list_ret_df_visu_cb.melt()
    list_ret_df_visu_cb = pd.DataFrame(list_ret_df_visu_cb.groupby('variable')['value'].sum())
    list_ret_df_visu_cb.reset_index(inplace=True)
    list_ret_df_visu_cb['corpus'] = 'Coral_brasil'
    
    #cesq
    
    list_ret_df_visu_cesq = list_ret_df.query('corpus in "Coral_esq"')
    list_ret_df_visu_cesq.drop(list_ret_df_visu_cesq.loc[:, 'retractings_full': 'retractings_syl'], axis = 1, inplace=True)
    # list_ret_df_visu = list_ret_df_visu_cesq.loc[:, 'oclus_des':]
    # list_ret_df_visu_cesq.drop('corpus', axis =1, inplace= True)
    list_ret_df_visu_cesq = list_ret_df_visu_cesq.melt()
    list_ret_df_visu_cesq = pd.DataFrame(list_ret_df_visu_cesq.groupby('variable')['value'].sum())
    list_ret_df_visu_cesq.reset_index(inplace=True)
    list_ret_df_visu_cesq['corpus'] = 'Coral_esq'
       
    list_ret_df_classe = pd.concat([list_ret_df_visu_cb, list_ret_df_visu_cesq], ignore_index=True)
    list_ret_df_classe.sort_values(by ='value', ascending = False, inplace=True)
    list_ret_df_classe.columns = ['Classe_do_segmento', 'Frequência', 'corpus']
    
    sns.set_style('whitegrid')
    plt.figure(figsize =(7, 5), dpi = 200)
    a = sns.barplot(data = list_ret_df_classe,x = 'Classe_do_segmento', y = 'Frequência', hue = 'corpus', estimator = sum )
    a.set_title('Classe do segmento inicial do retracting por corpora', fontsize = 18)
    a.set_ylabel('Quantidade', fontsize = 15)
    a.set_xlabel('Classe do segmento', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper right", frameon=True, fontsize=13)
    
else:
    list_ret_df_visu = list_ret_df.copy()
    list_ret_df_visu.drop(list_ret_df_visu.loc[:, 'retractings_full': 'retractings_syl'], axis = 1, inplace=True)
    # list_ret_df_visu = list_ret_df_visu.loc[:, 'oclus_des':]
    # list_ret_df_visu.drop('corpus', axis =1, inplace= True)
    list_ret_df_visu = list_ret_df_visu.melt()
    list_ret_df_visu = pd.DataFrame(list_ret_df_visu.groupby('variable')['value'].sum())
    list_ret_df_visu.reset_index(inplace=True)
    # list_ret_df_visu['corpus'] = 'Coral_brasil'
    list_ret_df_visu.columns = ['Classe_do_segmento', 'Frequência']
    
    sns.set_style('whitegrid')
    plt.figure(figsize =(7, 5), dpi = 200)
    a = sns.barplot(data = list_ret_df_visu.sort_values(by = 'Frequência', ascending = False),x = 'Classe_do_segmento', y = 'Frequência', palette = 'inferno')
    a.set_title('Classe do segmento inicial do retracting por corpora', fontsize = 18)
    a.set_ylabel('Quantidade', fontsize = 15)
    a.set_xlabel('Classe do segmento', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper right", frameon=True, fontsize=13)


#medidas
df_speech_m = round(df.groupby(['corpus', 'participant']).agg({'ut_length': ['sum', 'mean', 'std', 'min', 'max'], 'number_of_words':['sum', 'mean', 'std', 'min', 'max'], 'words_per_second': ['mean', 'min', 'max', 'std']}), 2)


try:
    
    df_inform_m = pd.DataFrame(df.groupby(['corpus','participant']).agg({'COB': ['sum', 'mean', 'max', 'min'], 'COM': ['sum', 'mean', 'max', 'min'], \
                                   'APC': ['sum', 'mean', 'max', 'min'], 'CMM': ['sum', 'mean', 'max', 'min'], \
                                    'CMM': ['sum', 'mean', 'max', 'min'], 'TOP_TPL': ['sum', 'mean', 'max', 'min'],\
                                    'APT': ['sum', 'mean', 'max', 'min'], 'PAR_PRL':['sum', 'mean', 'max', 'min'], \
                                    'INT': ['sum', 'mean', 'max', 'min'], 'SCA': ['sum', 'mean', 'max', 'min'], \
                                    'textual_units': ['sum', 'mean', 'max', 'min'], 'tonal_units': ['sum', 'mean', 'max', 'min'], \
                                    'AUX': ['sum', 'mean', 'max', 'min'], 'PHA': ['sum', 'mean', 'max', 'min'], \
                                    'ALL': ['sum', 'mean', 'max', 'min'], 'CNT': ['sum', 'mean', 'max', 'min'], \
                                    'DCT': ['sum', 'mean', 'max', 'min'], \
                                    'EXP': ['sum', 'mean', 'max', 'min']}))
except:
    pass 



df_inform_m = df_inform_m.round(2)

grouped_filt_patterns = df.groupby(['corpus','participant'])['filtered_patterns'].value_counts()



if "Coral_brasil" and "Coral_esq" in df.corpus.values:
    if df.textual_units.sum() > 0:
        
        counted_patterns = pd.DataFrame(df.groupby('corpus')['filtered_patterns'].value_counts())
        counted_patterns.columns = ['Frequência']
        counted_patterns.reset_index(inplace=True)
        counted_patterns.columns = ['corpus', 'Padrões_inf', 'Frequência']
        counted_patterns['Padrões_inf'] = counted_patterns['Padrões_inf'].str.replace('=', '').str.strip()
        counted_patterns['Padrões_inf']= counted_patterns['Padrões_inf'].apply(lambda x: '0' if len(x) < 3 else x)
        counted_patterns = counted_patterns.query('Padrões_inf != "0"')
      
        sns.set_style('whitegrid')
        plt.figure(dpi = 200, figsize = (9, 5))
        plt.xticks(rotation=90)
        b = sns.barplot(data = counted_patterns.sort_values(by = 'Frequência', ascending= False)[:15], x = 'Padrões_inf', y = "Frequência", hue = 'corpus')
        b.set_title('Padrões informacionais mais frequentes', fontsize = 16)
        b.set_ylabel("Frequência", fontsize = 15)
        b.set_xlabel('Padrões informacionais',fontsize=14)
        b.tick_params(labelsize=15)
        plt.legend(loc="upper right", frameon=True, fontsize=13)

else:     
    
    try: 
        
        counted_patterns = pd.DataFrame(df['filtered_patterns'].value_counts())
        counted_patterns.columns = ['Frequência']
        counted_patterns.reset_index(inplace=True)
        counted_patterns.columns = ['Padrões_inf', 'Frequência']
        counted_patterns = counted_patterns.query('Padrões_inf != ""')
        sns.set_style('whitegrid')
        plt.figure(dpi = 200, figsize = (6, 4))
        plt.xticks(rotation=90)
        b = sns.barplot(data = counted_patterns[:10], x = 'Padrões_inf', y = 'Frequência')
        b.set_title('Padrões informacionais mais frequentes', fontsize = 16)
        b.set_ylabel("Frequência", fontsize = 15)
        b.set_xlabel('Padrões informacionais',fontsize=14)
        b.tick_params(labelsize=15)
    except:
        pass 

print('By: José Carlos Costa \ Let me know if I can help you with something! \
      \n whatsapp: +55 31 98924 1307 \n \
      email: carlosjuniorcosta1@gmail.com')  

try:
    
    df1 = df.query('COB > 0')
    
    df_cb_bal = df1.query('corpus in "Coral_brasil"')
    df_cesq_bal = df1.query('corpus in "Coral_esq"')
except:
    pass

try:
    df_cb_cob1 = df_cb_bal.query('COB == 1')
    df_cb_cob2 = df_cb_bal.query('COB == 2')
    df_cb_cob3 = df_cb_bal.query('COB == 3')
    df_cb_cob4 = df_cb_bal.query('COB == 4')
    df_cb_cob5= df_cb_bal.query('COB == 5')
    df_cb_cob6 = df_cb_bal.query('COB == 6')
    df_cb_cob7 = df_cb_bal.query('COB == 7')
    df_cb_cob8 = df_cb_bal.query('COB == 8')
    df_cb_cob9 = df_cb_bal.query('COB == 9')
    df_cb_cob10 = df_cb_bal.query('COB == 10')
    df_cb_cob11 = df_cb_bal.query('COB == 11')
    df_cb_cob12 = df_cb_bal.query('COB == 12')
    df_cb_cob13 = df_cb_bal.query('COB == 13')
    df_cb_cob14 = df_cb_bal.query('COB == 14')
    df_cb_cob15 = df_cb_bal.query('COB == 15')
    
except:
    pass



try:
    df_cb_cob1 = df_cb_bal.query('COB == 1')
    df_cb_cob2 = df_cb_bal.query('COB == 2')
    df_cb_cob3 = df_cb_bal.query('COB == 3')
    df_cb_cob4 = df_cb_bal.query('COB == 4')
    df_cb_cob5= df_cb_bal.query('COB == 5')
    df_cb_cob6 = df_cb_bal.query('COB == 6')
    df_cb_cob7 = df_cb_bal.query('COB == 7')
    df_cb_cob8 = df_cb_bal.query('COB == 8')
    df_cb_cob9 = df_cb_bal.query('COB == 9')
    df_cb_cob10 = df_cb_bal.query('COB == 10')
    df_cb_cob11 = df_cb_bal.query('COB == 11')
    df_cb_cob12 = df_cb_bal.query('COB == 12')
    df_cb_cob13 = df_cb_bal.query('COB == 13')
    df_cb_cob14 = df_cb_bal.query('COB == 14')
    df_cb_cob15 = df_cb_bal.query('COB == 15')
    
except:
    pass



try:
    df_cesq_cob1 = df_cesq_bal.query('COB == 1')
    df_cesq_cob2 = df_cesq_bal.query('COB == 2')
    df_cesq_cob3 = df_cesq_bal.query('COB == 3')
    df_cesq_cob4 = df_cesq_bal.query('COB == 4')
    df_cesq_cob5= df_cesq_bal.query('COB == 5')
    df_cesq_cob6 = df_cesq_bal.query('COB == 6')
    df_cesq_cob7 = df_cesq_bal.query('COB == 7')
    df_cesq_cob8 = df_cesq_bal.query('COB == 8')
    df_cesq_cob9 = df_cesq_bal.query('COB == 9')
    df_cesq_cob10 = df_cesq_bal.query('COB == 10')
    df_cesq_cob11 = df_cesq_bal.query('COB == 11')
    df_cesq_cob12 = df_cesq_bal.query('COB == 12')
    df_cesq_cob13 = df_cesq_bal.query('COB == 13')
    df_cesq_cob14 = df_cesq_bal.query('COB == 14')
    df_cesq_cob15 = df_cesq_bal.query('COB == 15')
    
except:
    pass


if len(df_cb_cob1) and len(df_cesq_cob1) > 0:
   df_cob1_bal = pd.concat([df_cb_cob1, df_cesq_cob1])
   
   valor_cob1 = min(df_cob1_bal.corpus.value_counts())
   df_cob1_bal = df_cob1_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob1)))
   if len(df_cob1_bal) > 0:
       df_bal = df_cob1_bal.copy()
       
if len(df_cb_cob2) and len(df_cesq_cob2) > 0:
   df_cob2_bal = pd.concat([df_cb_cob2, df_cesq_cob2])
   
   valor_cob2 = min(df_cob2_bal.corpus.value_counts())
   df_cob2_bal = df_cob2_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob2)))
   if len(df_cob2_bal) > 0:
       df_bal = pd.concat([df_cob1_bal, df_cob2_bal])
       
if len(df_cb_cob3) and len(df_cesq_cob3) > 0:
   df_cob3_bal = pd.concat([df_cb_cob3, df_cesq_cob3])
   
   valor_cob3 = min(df_cob3_bal.corpus.value_counts())
   df_cob3_bal = df_cob3_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob3)))
   if len(df_cob3_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob3_bal])

if len(df_cb_cob4) and len(df_cesq_cob4) > 0:
   df_cob4_bal = pd.concat([df_cb_cob4, df_cesq_cob4])
   
   valor_cob4 = min(df_cob4_bal.corpus.value_counts())
   df_cob4_bal = df_cob4_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob4)))
   if len(df_cob4_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob4_bal])
       
     
if len(df_cb_cob5) and len(df_cesq_cob5) > 0:
   df_cob5_bal = pd.concat([df_cb_cob5, df_cesq_cob5])
   
   valor_cob5 = min(df_cob5_bal.corpus.value_counts())
   df_cob5_bal = df_cob5_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob5)))
   if len(df_cob5_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob5_bal])
       
       
if len(df_cb_cob6) and len(df_cesq_cob6) > 0:
   df_cob6_bal = pd.concat([df_cb_cob6, df_cesq_cob6])
   
   valor_cob6 = min(df_cob6_bal.corpus.value_counts())
   df_cob6_bal = df_cob6_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob6)))
   if len(df_cob6_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob6_bal])

if len(df_cb_cob7) and len(df_cesq_cob7) > 0:
   df_cob7_bal = pd.concat([df_cb_cob7, df_cesq_cob7])
   
   valor_cob7 = min(df_cob7_bal.corpus.value_counts())
   df_cob7_bal = df_cob7_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob7)))
   if len(df_cob7_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob7_bal])
       
       
if len(df_cb_cob8) and len(df_cesq_cob8) > 0:
   df_cob8_bal = pd.concat([df_cb_cob8, df_cesq_cob8])
   
   valor_cob8 = min(df_cob8_bal.corpus.value_counts())
   df_cob8_bal = df_cob8_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob8)))
   if len(df_cob8_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob8_bal])
       
if len(df_cb_cob9) and len(df_cesq_cob9) > 0:
   df_cob9_bal = pd.concat([df_cb_cob9, df_cesq_cob9])
   
   valor_cob9 = min(df_cob9_bal.corpus.value_counts())
   df_cob9_bal = df_cob9_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob9)))
   if len(df_cob9_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob9_bal])

if len(df_cb_cob10) and len(df_cesq_cob10) > 0:
   df_cob10_bal = pd.concat([df_cb_cob10, df_cesq_cob10])
   
   valor_cob10 = min(df_cob10_bal.corpus.value_counts())
   df_cob10_bal = df_cob10_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob10)))
   if len(df_cob10_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob10_bal])
       
if len(df_cb_cob11) and len(df_cesq_cob11) > 0:
   df_cob11_bal = pd.concat([df_cb_cob11, df_cesq_cob11])
   
   valor_cob11 = min(df_cob11_bal.corpus.value_counts())
   df_cob11_bal = df_cob11_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob11)))
   if len(df_cob11_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob11_bal])

if len(df_cb_cob12) and len(df_cesq_cob12) > 0:
   df_cob12_bal = pd.concat([df_cb_cob12, df_cesq_cob12])
   
   valor_cob12 = min(df_cob12_bal.corpus.value_counts())
   df_cob12_bal = df_cob12_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob12)))
   if len(df_cob12_bal) > 0:
       df_bal = pd.concat([df_bal, df_bal12_bal])
       
if len(df_cb_cob13) and len(df_cesq_cob13) > 0:
   df_cob13_bal = pd.concat([df_cb_cob13, df_cesq_cob13])
   
   valor_cob13 = min(df_cob13_bal.corpus.value_counts())
   df_cob13_bal = df_cob13_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob13)))
   if len(df_cob13_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob13_bal])
       
if len(df_cb_cob14) and len(df_cesq_cob14) > 0:
   df_cob14_bal = pd.concat([df_cb_cob14, df_cesq_cob14])
   
   valor_cob14 = min(df_cob14_bal.corpus.value_counts())
   df_cob14_bal = df_cob14_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob14)))
   if len(df_cob14_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob14_bal])
   
if len(df_cb_cob15) and len(df_cesq_cob15) > 0:
   df_cob15_bal = pd.concat([df_cb_cob15, df_cesq_cob15])
   
   valor_cob15 = min(df_cob15_bal.corpus.value_counts())
   df_cob15_bal = df_cob15_bal.groupby('corpus', group_keys = False).apply(lambda x: x.sample(min(len(x), valor_cob15)))
   if len(df_cob15_bal) > 0:
       df_bal = pd.concat([df_bal, df_cob15_bal])
   

df_bal = df_bal.query('COB < 6')
 
# df_bal = pd.read_csv('df_bal.csv')


if len(df_bal) > 0:
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'COM', hue = 'corpus', ci = 95)
    a.set_title('Comentários por amostra', fontsize = 18)
    a.set_ylabel('Média de Comentários', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'CMM', hue = 'corpus', ci = 95)
    a.set_title('Comentários Múltiplos por COB', fontsize = 18)
    a.set_ylabel('Média de Comentários Múltiplos', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))
      
   

    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'APC', hue = 'corpus', ci = 95)
    a.set_title('Apêndices de Comentário por COB', fontsize = 18)
    a.set_ylabel('Média de Apêndices de Comentários', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))

    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'CMM', hue = 'corpus', ci = 95)
    a.set_title('Comentários Múltiplos por COB', fontsize = 18)
    a.set_ylabel('Média de Apêndices de Comentários', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))








    sns.set_style('whitegrid')
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'TOP_TPL', hue = 'corpus', ci = 95)
    a.set_title('Tópicos por COB', fontsize = 18)
    a.set_ylabel('Média de Tópicos', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))
    

        
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'APT', hue = 'corpus', ci = 95)
    a.set_title('Apêndices de Tópico por COB', fontsize = 18)
    a.set_ylabel('Média de Apêndice de Tópicos', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))
    

    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'PAR_PRL', hue = 'corpus', ci = 95)
    a.set_title('Parentéticos por COB', fontsize = 18)
    a.set_ylabel('Média de Parentéticos', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'INT', hue = 'corpus', ci = 95)
    a.set_title('Introdutores Locutivos por COB', fontsize = 18)
    a.set_ylabel('Média de Introdutores Locutivos', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper right", frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))
    
        
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'inter_utterance', hue = 'corpus', ci = 95, estimator = np.mean)
    a.set_title('Enunciados interrompidos por COB', fontsize = 18)
    a.set_ylabel('Média de enunciados interrompidos', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper left", frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.5, 0.78, 0.5))
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'words_per_second', hue = 'corpus', ci = 95, estimator = np.mean)
    a.set_title('Palavras por segundo por COB', fontsize = 18)
    a.set_ylabel('Média de palavras por segundo', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper right", frameon=True, fontsize=13, bbox_to_anchor = (0.5, 0.5,0.78,0.5))
    
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'tonal_units', hue = 'corpus', ci = 95, estimator = np.mean)
    a.set_title('Unidades tonais por COB', fontsize = 18)
    a.set_ylabel('Média unidades tonais', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor = (0.5, 0.5,0.78,0.5))
    
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'number_of_words', hue = 'corpus', ci = 95, estimator = np.mean)
    a.set_title('Quantidade de palavras por COB', fontsize = 18)
    a.set_ylabel('Média palavras', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor = (0.5, 0.5,0.78,0.5))
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'ut_length', hue = 'corpus', ci = 95, estimator = np.mean)
    a.set_title('Duração das stanzas por COB', fontsize = 18)
    a.set_ylabel('Média de duração', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13, bbox_to_anchor = (0.5, 0.5,0.78,0.5))
    
    padrao_enu_int = df_bal.query('inter_utterance > 0')


    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = padrao_enu_int, x = 'COB', y = 'textual_units', hue = 'corpus', ci = 95, estimator = np.mean)
    a.set_title('Unidades textuais em enunciados interrompidos', fontsize = 18)
    a.set_ylabel('Média de Unidades Textuais', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13,  bbox_to_anchor = (0.5, 0.5,0.78,0.5))
    
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'words_per_second', hue = 'corpus', ci = 95, estimator = np.mean)
    a.set_title('Palavras por segundo por COB', fontsize = 18)
    a.set_ylabel('Média de palavras por segundo', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(loc="upper right", frameon=True, fontsize=13, bbox_to_anchor = (0.5, 0.5,0.78,0.5))
    
    plt.figure(figsize =(8, 5), dpi = 200)
    a = sns.barplot(data = df_bal, x = 'COB', y = 'AUX', hue = 'corpus', ci = 95)
    a.set_title('Unidades Dialógicas por COB', fontsize = 18)
    a.set_ylabel('Média de Unidades Dialógicas', fontsize = 15)
    a.set_xlabel('COBs', fontsize = 15)
    a.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.legend(frameon=True, fontsize=13,  bbox_to_anchor = (0.5, 0.5,0.78,0.5))
    
      

try: 
    if len(df_bal) > 0:
        from scipy.stats import mannwhitneyu
        if len(df_cob1_bal)> 0:
            
            lista_comparacao = ['COM', 'APC', 'CMM', 'TOP_TPL', 'APT', 'PAR_PRL', 'INT', 'SCA',
                   'textual_units', 'tonal_units', 'AUX','inter_utterance', 'number_of_words','words_per_second', 
                   'retractings', 'retr_words']
            
            corpus_cesq_1COB= df_bal.query("corpus in 'Coral_esq' and COB == 1")
            
            corpus_cb_1COB = df_bal.query("corpus in 'Coral_brasil' and COB == 1")
            
            
            # corpus_shapiro_1COB = df_bal.query("COB in '1'")
             
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_1COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_1COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_1_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_1_cob = pd.DataFrame([test_1_cob]).transpose()
            test_1_cob.reset_index(inplace=True)
            test_1_cob.columns = ['variavel', 'p_value']
            test_1_cob['p_value'] = test_1_cob['p_value']
           
            if len(test_1_cob) > 0:
                test_1_cob['COB'] = 1
                test_p_value= test_1_cob.copy()
                s_pvalue_1_cob = test_1_cob.query('p_value < 0.05')
            
               
        if len(df_cob2_bal) > 0:
            
            corpus_cesq_2COB= df_bal.query("corpus in 'Coral_esq' and COB == 2")
            
            corpus_cb_2COB = df_bal.query("corpus in 'Coral_brasil' and COB == 2")
            
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_2COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_2COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_2_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_2_cob = pd.DataFrame([test_2_cob]).transpose()
            test_2_cob.reset_index(inplace=True)
            test_2_cob.columns = ['variavel', 'p_value']
            test_2_cob['p_value'] = test_2_cob['p_value']
            if len(test_2_cob) > 0:
                test_2_cob['COB'] = 2
                test_p_value = pd.concat([test_p_value, test_2_cob])
                s_pvalue_2_cob = test_2_cob.query('p_value < 0.05')
                
            
        if len(df_cob3_bal) > 0:
                    
            corpus_cesq_3COB= df_bal.query("corpus in 'Coral_esq' and COB == 3")
            
            corpus_cb_3COB = df_bal.query("corpus in 'Coral_brasil' and COB == 3")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_3COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_3COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_3_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_3_cob = pd.DataFrame([test_3_cob]).transpose()
            test_3_cob.reset_index(inplace=True)
            test_3_cob.columns = ['variavel', 'p_value']
            test_3_cob['p_value'] = test_3_cob['p_value']
            if len(test_3_cob) > 0:
                test_3_cob['COB'] = 3
                test_p_value = pd.concat([test_p_value, test_3_cob])
                s_pvalue_3_cob = test_3_cob.query('p_value < 0.05')
            
        if len(df_cob4_bal) > 0:
            
                    
            
            corpus_cesq_4COB= df_bal.query("corpus in 'Coral_esq' and COB == 4")
            
            corpus_cb_4COB = df_bal.query("corpus in 'Coral_brasil' and COB == 4")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_4COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_4COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_4_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_4_cob = pd.DataFrame([test_4_cob]).transpose()
            test_4_cob.reset_index(inplace=True)
            test_4_cob.columns = ['variavel', 'p_value']
            test_4_cob['p_value'] = test_4_cob['p_value']
            if len(test_4_cob) > 0:
                test_4_cob['COB'] = 4
                test_p_value = pd.concat([test_p_value, test_4_cob])
                s_pvalue_4_cob = test_4_cob.query('p_value < 0.05')
            
        if len(df_cob5_bal) > 0:
                    
            corpus_cesq_5COB= df_bal.query("corpus in 'Coral_esq' and COB == 5")
            
            corpus_cb_5COB = df_bal.query("corpus in 'Coral_brasil' and COB == 5")
            
            
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_5COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_5COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_5_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_5_cob = pd.DataFrame([test_5_cob]).transpose()
            test_5_cob.reset_index(inplace=True)
            test_5_cob.columns = ['variavel', 'p_value']
            test_5_cob['p_value'] = test_5_cob['p_value']
            if len(test_5_cob) > 0:
                test_5_cob['COB'] = 5
                test_p_value = pd.concat([test_p_value, test_5_cob])
                    
            
                s_pvalue_5_cob = test_5_cob.query('p_value < 0.05')
            
            
        if len(df_cob6_bal) > 0:
                    
            corpus_cesq_6COB= df_bal.query("corpus in 'Coral_esq' and COB == 6")
            
            corpus_cb_6COB = df_bal.query("corpus in 'Coral_brasil' and COB == 6")
            
            
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_6COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_6COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_6_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_6_cob = pd.DataFrame([test_6_cob]).transpose()
            test_6_cob.reset_index(inplace=True)
            test_6_cob.columns = ['variavel', 'p_value']
            test_6_cob['p_value'] = test_6_cob['p_value']
            if len(test_6_cob) > 0:
                test_6_cob['COB'] = 6
                test_p_value = pd.concat([test_p_value, test_6_cob])
                s_pvalue_6_cob = test_6_cob.query('p_value < 0.05')
        
        
        if len(df_cob7_bal) > 0:
                    
            corpus_cesq_7COB= df_bal.query("corpus in 'Coral_esq' and COB == 7")
            
            corpus_cb_7COB = df_bal.query("corpus in 'Coral_brasil' and COB == 7")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_7COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_7COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_7_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_7_cob = pd.DataFrame([test_7_cob]).transpose()
            test_7_cob.reset_index(inplace=True)
            test_7_cob.columns = ['variavel', 'p_value']
            test_7_cob['p_value'] = test_7_cob['p_value']
            if len(test_7_cob) > 0:
                test_7_cob['COB'] = 7
                test_p_value = pd.concat([test_p_value, test_7_cob])
                s_pvalue_7_cob = test_7_cob.query('p_value < 0.05')
            
         
        if len(df_cob8_bal) > 0:
                    
            corpus_cesq_8COB= df_bal.query("corpus in 'Coral_esq' and COB == 8")
            
            corpus_cb_8COB = df_bal.query("corpus in 'Coral_brasil' and COB == 8")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_8COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_8COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_8_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_8_cob = pd.DataFrame([test_8_cob]).transpose()
            test_8_cob.reset_index(inplace=True)
            test_8_cob.columns = ['variavel', 'p_value']
            test_8_cob['p_value'] = test_8_cob['p_value']
            
            if len(test_8_cob) > 0:
                test_8_cob['COB'] = 8
                test_p_value = pd.concat([test_p_value, test_8_cob])
                s_pvalue_8_cob = test_8_cob.query('p_value < 0.05')  
            
            
        
        if len(df_cob9_bal) > 0:
                    
            corpus_cesq_9COB= df_bal.query("corpus in 'Coral_esq' and COB == 9")
            
            corpus_cb_9COB = df_bal.query("corpus in 'Coral_brasil' and COB == 9")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_9COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_9COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_9_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_9_cob = pd.DataFrame([test_9_cob]).transpose()
            test_9_cob.reset_index(inplace=True)
            test_9_cob.columns = ['variavel', 'p_value']
            test_9_cob['p_value'] = test_9_cob['p_value']
            if len(test_9_cob) > 0:
                
                test_9_cob['COB'] = 9
                test_p_value = pd.concat([test_p_value, test_9_cob])                
                s_pvalue_9_cob = test_9_cob.query('p_value < 0.05')  
            
            
        if len(df_cob10_bal) > 0:
                    
            corpus_cesq_10COB= df_bal.query("corpus in 'Coral_esq' and COB == 10")
            
            corpus_cb_10COB = df_bal.query("corpus in 'Coral_brasil' and COB == 10")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_10COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_10COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_10_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_10_cob = pd.DataFrame([test_10_cob]).transpose()
            test_10_cob.reset_index(inplace=True)
            test_10_cob.columns = ['variavel', 'p_value']
            test_10_cob['p_value'] = test_10_cob['p_value']
            if len(test_10_cob) > 0:
                test_10_cob['COB'] = 10
                test_p_value = pd.concat([test_p_value, test_10_cob])                
                s_pvalue_10_cob = test_10_cob.query('p_value < 0.05')  
            
    
        if len(df_cob11_bal) > 0:
                    
            corpus_cesq_11COB= df_bal.query("corpus in 'Coral_esq' and COB == 11")
            
            corpus_cb_11COB = df_bal.query("corpus in 'Coral_brasil' and COB == 11")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_11COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_11COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_11_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_11_cob = pd.DataFrame([test_11_cob]).transpose()
            test_11_cob.reset_index(inplace=True)
            test_11_cob.columns = ['variavel', 'p_value']
            test_11_cob['p_value'] = test_11_cob['p_value']
            if len(test_11_cob) > 0:
                test_11_cob['COB'] = 11
                test_p_value = pd.concat([test_p_value, test_11_cob])
                s_pvalue_11_cob = test_11_cob.query('p_value < 0.05') 
        
        

        
        if len(df_cob12_bal) > 0:
                    
            corpus_cesq_12COB= df_bal.query("corpus in 'Coral_esq' and COB == 12")
            
            corpus_cb_12COB = df_bal.query("corpus in 'Coral_brasil' and COB == 12")
                 
            lista_resultado = []
            lista_etiquetas = []
            for coluna_cb, linha_cb in corpus_cb_12COB.iteritems():
                for coluna_cesq, linha_cesq in corpus_cesq_12COB.iteritems():
                    if coluna_cb in lista_comparacao:
                        if coluna_cesq == coluna_cb:
                            lista_etiquetas.append(coluna_cesq)
                            lista_resultado.append(mannwhitneyu(linha_cb, linha_cesq, alternative='two-sided')[1])
            
            test_12_cob = dict(zip(lista_etiquetas, lista_resultado))     
            test_12_cob = pd.DataFrame([test_12_cob]).transpose()
            test_12_cob.reset_index(inplace=True)
            test_12_cob.columns = ['variavel', 'p_value']
            test_12_cob['p_value'] = test_12_cob['p_value']
            if len(test_12_cob) > 0:
                test_12_cob['COB'] = 12
                test_p_value = pd.concat([test_p_value, test_12_cob])
                s_pvalue_12_cob = test_12_cob.query('p_value < 0.05') 
        test_p_value.to_csv('test_p_value.csv')
        test_p_value.to_excel('test_p_value.xlsx')
except:
    pass


try:
        
    if len(test_p_value) > 0:
            
        test_p_value = test_p_value.query('p_value <= 0.05')
        
        plt.figure(figsize =(8, 5), dpi = 200)
        a = sns.barplot(data = test_p_value, x = 'COB', y = 'p_value', hue = 'variavel', palette = 'inferno')
        a.set_title('Resultados estatisticamente relevantes', fontsize = 18)
        a.set_ylabel('p-value - log', fontsize = 15)
        a.set_xlabel('COBs', fontsize = 15)
        a.tick_params(labelsize = 15)
        a.set_yscale("log")
        plt.xticks(rotation = 90)
        plt.legend(frameon=True, fontsize=13, bbox_to_anchor = (0.5, 0.5,0.85,0.5))
except:
    pass



if "=PAUSA_P=" in df.utterances.values:
    try:
        
        df['TMT'] = df['utterances'].apply(lambda x: len(re.findall(r"=PAUSA_P=?", x)))
        #df['corpus'] = df['audio'].apply(lambda x: "Coral_esq" if x.startswith('m') else 'Coral_brasil')
        
        df['word_before']= df['utterances_POS'].apply(lambda x: ' '.join(re.findall(r"(?<=\[)\('(\w+)", x)))
        df['class_before'] = df['utterances_POS'].apply(lambda x: ' '.join(re.findall(r"(?<=\[)\('\w+',\s'(\w+)", x)))
        
        
        pausas_index = pd.Series((df.index[df['utterances'] == '=PAUSA_P=']))
        pausas_mais_um = pd.Series((df.index[df['utterances'] == '=PAUSA_P='] + 1))
        
        valores = df['ut_length'].iloc[pausas_index]
        
        
        df['pause_len'] = valores.set_axis(pausas_mais_um)
        df['pause_len'] = df['pause_len'].apply(lambda x: 0 if pd.isnull(x) == True else x)
        
        df_pauses = df.query('pause_len > 0')
        
        df_pauses = df_pauses.query('word_before != "PAUSA_P"')
        df_pauses['word_before'] = df_pauses['word_before'].apply(lambda x: '0' if len(x) < 1 else x)
        df_pauses['class_before'] = df_pauses['class_before'].apply(lambda x: '0' if len(x) < 1 else x)
        
        df_pauses = df_pauses.query('word_before != "0"')
        df_pauses = df_pauses.query('class_before != "0"')
        
        # df.to_csv('df_total.csv')
        # df_pauses.to_csv('df_pausas.csv')
        
        freq_dist = pd.DataFrame(df_pauses['word_before'].value_counts())
        freq_dist.reset_index(inplace=True)
        freq_dist.columns = ['palavra_diante_pausa', 'frequência']
        freq_dist.to_csv('df_palavras_diante.csv')
        
        sns.set_style('whitegrid')
        plt.figure(dpi = 300, figsize=(7, 5))
        a = sns.histplot(data= df_pauses, x = 'pause_len', kde = True)
        a.set_title('Distribuição da duração das pausas', fontsize = 16)
        a.set_xlabel("Tempo - (s)",fontsize= 14)
        a.set_ylabel("Frequência",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.show()
        
        sns.set_style('whitegrid')
        plt.figure(dpi = 200, figsize=(8, 5))
        a = sns.barplot(data = df_pauses , x = 'class_before', y = 'pause_len',  palette = 'inferno', estimator = np.mean, ci = False)
        a.set_title('Duração de pausas diante de classes de palavras', fontsize = 16)
        a.set_xlabel("Classes de palavras",fontsize= 14)
        a.set_ylabel("Média de duração",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.xticks(rotation=90)
        plt.show()
        
        
        plt.figure(dpi = 200, figsize=(8, 5))
        a = sns.countplot(data = df_pauses, x = 'class_before', palette = 'inferno')
        a.set_title('Frequência de pausas diante de classes de palavras', fontsize = 16)
        a.set_xlabel("Classes de palavras",fontsize= 14)
        a.set_ylabel("Frequência",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.xticks(rotation=90)
        plt.show()
        
        
        plt.figure(dpi = 200, figsize=(8, 5))
        a = sns.lineplot(data = freq_dist[:30], x = 'palavra_diante_pausa', y = 'frequência', palette = 'inferno')
        a.set_title('Palavras mais frequentes diante de pausas preenchidas', fontsize = 16)
        a.set_xlabel("Classes de palavras",fontsize= 14)
        a.set_ylabel("Frequência",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.xticks(rotation=90)
        plt.show()
        
    except: 
        pass
#daqui 

try: 
    
    if len(df_bal) > 0:        
            
        if "Coral_brasil" and "Coral_esq" in df_bal.corpus.values:
            df_bal_cb = df_bal.query('corpus in "Coral_brasil"')
        
            dist_ret_cb = df_bal_cb['dist_retractings'].tolist()
            list_participants_cb = df_bal_cb['participant'].to_list()
          
            list_ret_cb= []
            
            for x in dist_ret_cb:
                for y in x.split():
                    list_ret_cb.append(y)
           
            list_ret_cb = '\n'.join(list_ret_cb)
            list_ret_cb = re.sub('\[|\]|\"|,', '', list_ret_cb)
            
            
            list_ret_cb = [x.lower() for x in list_ret_cb.splitlines() if len(x) > 0]
            
            list_ret_df_bal_cb = pd.DataFrame([x.strip() for x in list_ret_cb], columns=['retractings_full'])
            list_ret_df_bal_cb['corpus'] = 'Coral_brasil'
            
            #cesq
            
            df_bal_cesq = df_bal.query('corpus in "Coral_esq"')
        
            dist_ret_cesq = df_bal_cesq['dist_retractings'].tolist()
            list_participants_cesq = df_bal_cesq['participant'].to_list()
          
            list_ret_cesq= []
            
            for x in dist_ret_cesq:
                for y in x.split():
                    list_ret_cesq.append(y)
           
            list_ret_cesq = '\n'.join(list_ret_cesq)
            list_ret_cesq = re.sub('\[|\]|\"|,', '', list_ret_cesq)
           
            list_ret_cesq = [x.lower() for x in list_ret_cesq.splitlines() if len(x) > 0]
            
            list_ret_df_bal_cesq = pd.DataFrame([x.strip() for x in list_ret_cesq], columns=['retractings_full'])
            list_ret_df_bal_cesq['corpus'] = 'Coral_esq'
            
            list_ret_df_bal = pd.concat([list_ret_df_bal_cb, list_ret_df_bal_cesq])
            
        else:
          
            dist_ret = df_bal['dist_retractings'].tolist()
            list_participants = df_bal['participant'].to_list()
          
            list_ret= []
            
            for x in dist_ret:
                for y in x.split():
                    list_ret.append(y)
           
            list_ret = '\n'.join(list_ret)
            list_ret = re.sub('\[|\]|\"|,', '', list_ret)
            
            
            list_ret = [x.lower() for x in list_ret.splitlines() if len(x) > 0]
            
            list_ret_df_bal = pd.DataFrame([x.strip() for x in list_ret], columns=['retractings_full'])
            
        list_ret_df_bal['retractings'] = list_ret_df_bal['retractings_full'].apply(lambda x: re.sub(r"&|-|<|>|\=", '', x))
        list_ret_df_bal['retractings_syl'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r".?uai.?|.?uão.?|.?ai.?|.?ói.?|.?ua.?|.?uo.?|.?io.?|.?ió.?|.?éi.?|.?ei.?|.?ie.?|.?ói.?|.?oi.?|.?au.?|.?ou.?|.?éu.?|.?ui.?|.?a.?|.?á.?|.?â.?|.?ã.?|.?é.?|.?ê.?|.?e.?|.?o.?|.?ô.?|.?õ.?|.?ó.?|.?ò.?|.?i.?|.?í.?",  str(x), flags = re.IGNORECASE)))
        
        list_ret_df_bal['oclus_des'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r"^p|^k|^k$|^c$|^ca|^co|^cu|^cã|^cô|^câ|^cõ|^cú|^q.?.?|^cr|^te(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|u|ú|ó|o|ô|õ)|^té|^tê|^ta|^tá|^tã|^tâ|^to|^tô|^tu|^tú|^tr", x)))
        list_ret_df_bal['oclus_voz'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r"^b|^ga|^go|^gu|^gão|^gá|^gó|^gú|^gâ|^gr|^g$|^da|^dá|^dã|^dõ|^dô|^dú|^du|^do|^dr|^de(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)", x)))
        list_ret_df_bal['laterais'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r"(^l|^lh)", x)))
        list_ret_df_bal['fricativas_des'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r"(^s|^x|^r|^ch|^f|^ce|^ci|^cê|^cí|^cé)", x)))
        list_ret_df_bal['fricativas_voz'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r'(^z|^j|^v|^ge|^gi)', x)))
        list_ret_df_bal['oclus_nas'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r'(^m|^n)', x)))
        list_ret_df_bal['africadas_des'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r'^te(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|a|á|ã|à|â|e|é|ê|i|í|o|ó|õ|ô|u|ú)|^ti|^t$', x)))
        list_ret_df_bal['africadas_voz'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r'^de(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)|^di|^d$', x)))
        list_ret_df_bal['vogais'] = list_ret_df_bal['retractings'].apply(lambda x: ' '.join(re.findall(r"^a|^e|^i|^o|^u|^ã|^õ|^à|^á|^â|^ô|^ó|^é|^ê|^ú|^h", x)))
        
        list_ret_df_bal = list_ret_df_bal.loc[list_ret_df_bal['retractings_full'] != 'xxx']
        list_ret_df_bal = list_ret_df_bal.loc[list_ret_df_bal['retractings_full'] != 'xxxx']
        list_ret_df_bal = list_ret_df_bal.loc[list_ret_df_bal['retractings_full'] != 'yyy']
        list_ret_df_bal = list_ret_df_bal.loc[list_ret_df_bal['retractings_full'] != 'yyyy']
        list_ret_df_bal = list_ret_df_bal.loc[list_ret_df_bal['retractings_full'] != 'hhh']
        
        list_ret_df_bal['retractings'] = list_ret_df_bal['retractings'].apply(lambda x: x[:3] if len(x) > 3 else x)
        
        list_ret_df_bal['oclus_des'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r"^p|^k|^k$|^c$|^ca|^co|^cu|^cã|^cô|^câ|^cõ|^cú|^q.?.?|^cr|^te(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|u|ú|ó|o|ô|õ)|^té|^tê|^ta|^tá|^tã|^tâ|^to|^tô|^tu|^tú|^tr", str(x))))
        list_ret_df_bal['oclus_voz'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r"^b|^ga|^go|^gu|^gão|^gá|^gó|^gú|^gâ|^gr|^g$|^da|^dá|^dã|^dõ|^dô|^dú|^du|^do|^dr|^de(?=p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)", x)))
        list_ret_df_bal['laterais'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r"(^l|^lh)", x)))
        list_ret_df_bal['fricativas_des'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r"(^s|^x|^r|^ch|^f|^ce|^ci|^cê|^cí|^cé)", x)))
        list_ret_df_bal['fricativas_voz'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r'(^z|^j|^v|^ge|^gi)', x)))
        list_ret_df_bal['oclus_nas'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r'(^m|^n)', x)))
        list_ret_df_bal['africadas_des'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r'^te(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|a|á|ã|à|â|e|é|ê|i|í|o|ó|õ|ô|u|ú)|^ti|^t$', x)))
        list_ret_df_bal['africadas_voz'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r'^de(?!p|t|k|q|c|f|s|ss|x|ch|l|r|rr|b|d|g|v|z|j|m|n|i|u|a|á|ã|â|à|e|é|ê|i|í|ó|ô|õ|u|ú)|^di|^d$', x)))
        list_ret_df_bal['vogais'] = list_ret_df_bal['retractings'].apply(lambda x: len(re.findall(r"^a|^e|^i|^o|^u|^ã|^õ|^à|^á|^â|^ô|^ó|^é|^ê|^ú|^h", x)))
        
        
        if "Coral_brasil" and "Coral_esq" in df_bal.corpus.values:
            retracted_words_bal = pd.DataFrame(list_ret_df_bal.groupby('corpus')['retractings_full'].value_counts())
            retracted_words_bal.columns = ['Frequência']
            retracted_words_bal.reset_index(inplace=True)
            retracted_words_bal.columns = ['corpus', 'retractings', 'Frequência'] 
        else:   
            
            contagem_retratadas = pd.DataFrame(list_ret_df_bal['retractings_full'].value_counts())
            contagem_retratadas.reset_index(inplace=True)
            contagem_retratadas.columns = ['retractings', 'Frequência']  
        
        if "Coral_brasil" and "Coral_esq" in df_bal.corpus.values:
            plt.figure(figsize =(9, 5), dpi = 200)
            a = sns.barplot(data = retracted_words.sort_values(by = 'Frequência', ascending =False)[:30], x = 'retractings', y = 'Frequência', hue = 'corpus')
            a.set_title('Palavras retratadas nos corpora', fontsize = 18)
            a.set_ylabel('Quantidade', fontsize = 15)
            a.set_xlabel('Classe do segmento', fontsize = 15)
            a.tick_params(labelsize = 15)
            plt.xticks(rotation = 90)
            plt.legend(loc="upper right", frameon=True, fontsize=13)
            
        else:
            plt.figure(figsize =(9, 5), dpi = 200)
            a = sns.barplot(data = contagem_retratadas[:30], x = 'retractings', y = 'Frequência')
            a.set_title('Frequência de palavras retratadas', fontsize = 18)
            a.set_ylabel('Quantidade', fontsize = 15)
            a.set_xlabel('Classe do segmento', fontsize = 15)
            a.tick_params(labelsize = 15)
            plt.xticks(rotation = 90)
            plt.legend(loc="upper right", frameon=True, fontsize=13)
            
        if "Coral_brasil" and "Coral_esq" in df_bal.corpus.values:
            list_ret_df_bal_visu_cb = list_ret_df_bal.query('corpus in "Coral_brasil"')
            list_ret_df_bal_visu_cb.drop(list_ret_df_bal_visu_cb.loc[:, 'retractings_full': 'retractings_syl'], axis = 1, inplace=True)
            # list_ret_df_bal_visu = list_ret_df_bal_visu_cb.loc[:, 'oclus_des':]
            # list_ret_df_bal_visu_cb.drop('corpus', axis =1, inplace= True)
            list_ret_df_bal_visu_cb = list_ret_df_bal_visu_cb.melt()
            list_ret_df_bal_visu_cb = pd.DataFrame(list_ret_df_bal_visu_cb.groupby('variable')['value'].sum())
            list_ret_df_bal_visu_cb.reset_index(inplace=True)
            list_ret_df_bal_visu_cb['corpus'] = 'Coral_brasil'
            
            #cesq
            
            list_ret_df_bal_visu_cesq = list_ret_df_bal.query('corpus in "Coral_esq"')
            list_ret_df_bal_visu_cesq.drop(list_ret_df_bal_visu_cesq.loc[:, 'retractings_full': 'retractings_syl'], axis = 1, inplace=True)
            # list_ret_df_bal_visu = list_ret_df_bal_visu_cesq.loc[:, 'oclus_des':]
            # list_ret_df_bal_visu_cesq.drop('corpus', axis =1, inplace= True)
            list_ret_df_bal_visu_cesq = list_ret_df_bal_visu_cesq.melt()
            list_ret_df_bal_visu_cesq = pd.DataFrame(list_ret_df_bal_visu_cesq.groupby('variable')['value'].sum())
            list_ret_df_bal_visu_cesq.reset_index(inplace=True)
            list_ret_df_bal_visu_cesq['corpus'] = 'Coral_esq'
               
            list_ret_df_bal_classe = pd.concat([list_ret_df_bal_visu_cb, list_ret_df_bal_visu_cesq], ignore_index=True)
            list_ret_df_bal_classe.sort_values(by ='value', ascending = False, inplace=True)
            list_ret_df_bal_classe.columns = ['Classe_do_segmento', 'Frequência', 'corpus']
            
            sns.set_style('whitegrid')
            plt.figure(figsize =(7, 5), dpi = 200)
            a = sns.barplot(data = list_ret_df_bal_classe,x = 'Classe_do_segmento', y = 'Frequência', hue = 'corpus', estimator = sum )
            a.set_title('Classe do segmento inicial do retracting por corpora', fontsize = 18)
            a.set_ylabel('Quantidade', fontsize = 15)
            a.set_xlabel('Classe do segmento', fontsize = 15)
            a.tick_params(labelsize = 15)
            plt.xticks(rotation = 90)
            plt.legend(loc="upper right", frameon=True, fontsize=13)
            
        else:
            list_ret_df_bal_visu = list_ret_df_bal.copy()
            list_ret_df_bal_visu.drop(list_ret_df_bal_visu.loc[:, 'retractings_full': 'retractings_syl'], axis = 1, inplace=True)
            # list_ret_df_bal_visu = list_ret_df_bal_visu.loc[:, 'oclus_des':]
            # list_ret_df_bal_visu.drop('corpus', axis =1, inplace= True)
            list_ret_df_bal_visu = list_ret_df_bal_visu.melt()
            list_ret_df_bal_visu = pd.DataFrame(list_ret_df_bal_visu.groupby('variable')['value'].sum())
            list_ret_df_bal_visu.reset_index(inplace=True)
            # list_ret_df_bal_visu['corpus'] = 'Coral_brasil'
            list_ret_df_bal_visu.columns = ['Classe_do_segmento', 'Frequência']
            
            sns.set_style('whitegrid')
            plt.figure(figsize =(7, 5), dpi = 200)
            a = sns.barplot(data = list_ret_df_bal_visu.sort_values(by = 'Frequência', ascending = False),x = 'Classe_do_segmento', y = 'Frequência', palette = 'inferno')
            a.set_title('Classe do segmento inicial do retracting', fontsize = 18)
            a.set_ylabel('Quantidade', fontsize = 15)
            a.set_xlabel('Classe do segmento', fontsize = 15)
            a.tick_params(labelsize = 15)
            plt.xticks(rotation = 90)
            plt.legend(loc="upper right", frameon=True, fontsize=13)


except:
    pass 



if "Coral_brasil" and "Coral_esq" in df_bal.corpus.values:
    if df_bal.textual_units.sum() > 0:
        
        counted_patterns = pd.DataFrame(df_bal.groupby('corpus')['filtered_patterns'].value_counts())
        counted_patterns.columns = ['Frequência']
        counted_patterns.reset_index(inplace=True)
        counted_patterns.columns = ['corpus', 'Padrões_inf', 'Frequência']
        counted_patterns['Padrões_inf'] = counted_patterns['Padrões_inf'].str.replace('=', '').str.strip()
        counted_patterns['Padrões_inf']= counted_patterns['Padrões_inf'].apply(lambda x: '0' if len(x) < 3 else x)
        counted_patterns = counted_patterns.query('Padrões_inf != "0"')
      
        sns.set_style('whitegrid')
        plt.figure(dpi = 200, figsize = (9, 5))
        plt.xticks(rotation=90)
        b = sns.barplot(data = counted_patterns.sort_values(by = 'Frequência', ascending= False)[:15], x = 'Padrões_inf', y = "Frequência", hue = 'corpus')
        b.set_title('Padrões informacionais mais frequentes nas amostras balanceadas', fontsize = 16)
        b.set_ylabel("Frequência", fontsize = 15)
        b.set_xlabel('Padrões informacionais',fontsize=14)
        b.tick_params(labelsize=15)
        plt.legend(loc="upper right", frameon=True, fontsize=13)


if "Coral_brasil" and "Coral_esq" in df.corpus.values:
    df.to_csv('df_all_data.csv')
    df_bal.to_csv('df_bal.csv')
    df_speech_m.to_csv('df_speech_m.csv')
    df_inform_m.to_csv('df_textual_units_per_participant.csv')
    grouped_filt_patterns.to_csv('df_patterns_participants.csv')
    list_ret_df_bal.to_csv('df_retractings_bal.csv')
    retracted_words_bal.to_csv('df_retracted_words_count_bal.csv')
    retracted_words.to_csv('df_retracted_words_count_all.csv')

else:
    df.to_csv('df_all_data.csv')
    df_speech_m.to_csv('df_speech_m.csv')
    df_inform_m.to_csv('df_textual_units_per_participant.csv')
    grouped_filt_patterns.to_csv('df_patterns_participants.csv')
    list_ret_df.to_csv('df_retractings.csv')
    contagem_retratadas.to_csv('df_retracted_words_count.csv')  


print('Let me know if I can help you with something! \n by: José Carlos Costa \n email:carlosjuniorcosta1@gmail.com')