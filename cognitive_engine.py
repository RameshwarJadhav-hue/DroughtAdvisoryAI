"""
Cognitive Reasoning and Farmer Advisory Engine
Combines machine learning outputs with explainable symbolic reasoning
and bilingual (English/Marathi) agricultural advisory rules.
"""

def generate_cognitive_reasoning(deficit, soil, groundwater, temp, prediction, lang="en"):
    """
    Generates explainable causal reasoning describing WHY the AI system
    arrived at the specific drought risk classification.
    """
    reasons_en = []
    reasons_mr = []

    # 1. Rainfall Deficit Evaluation
    if deficit >= 40:
        reasons_en.append(f"Severe rainfall deficit ({deficit:.1f}%), triggering official Trigger-1 drought warning.")
        reasons_mr.append(f"पावसात अत्यंत गंभीर तूट ({deficit:.1f}%), ज्यामुळे अधिकृत 'ट्रिगर-१' दुष्काळ निकष लागू होतात.")
    elif deficit >= 30:
        reasons_en.append(f"Substantial rainfall deficit ({deficit:.1f}%), indicating significant monsoon shortfall.")
        reasons_mr.append(f"पावसात मोठी तूट ({deficit:.1f}%), मान्सूनच्या कमतरतेचे स्पष्ट संकेत.")
    elif deficit >= 20:
        reasons_en.append(f"Moderate rainfall deficit ({deficit:.1f}%), requiring cautious water budgeting.")
        reasons_mr.append(f"पावसात मध्यम तूट ({deficit:.1f}%), पाण्याचे काटेकोर नियोजन आवश्यक.")
    else:
        reasons_en.append(f"Rainfall levels near normal deficit ({deficit:.1f}%).")
        reasons_mr.append(f"पावसाचे प्रमाण समाधानकारक/कमी तूट ({deficit:.1f}%).")

    # 2. Soil Moisture Evaluation
    if soil < 22:
        reasons_en.append(f"Critical soil moisture stress ({soil:.1f}% < 22%), severely impeding root absorption.")
        reasons_mr.append(f"मातीतील ओलावा धोक्याच्या पातळीखाली ({soil:.1f}% < २२%), पिकांच्या मुळांना तीव्र ताण.")
    elif soil < 30:
        reasons_en.append(f"Depleted soil moisture ({soil:.1f}%), approaching moisture deficit threshold.")
        reasons_mr.append(f"मातीतील ओलावा कमी ({soil:.1f}%), लवकरच संरक्षित सिंचनाची गरज.")
    else:
        reasons_en.append(f"Adequate root-zone soil moisture available ({soil:.1f}%).")
        reasons_mr.append(f"जमिनीत पिकांसाठी पुरेसा ओलावा उपलब्ध ({soil:.1f}%).")

    # 3. Groundwater Depth Evaluation
    if groundwater >= 25.0:
        reasons_en.append(f"Deep groundwater table ({groundwater:.1f}m below ground), severely limiting borewell recharge.")
        reasons_mr.append(f"भूजल पातळी खोल गेली आहे ({groundwater:.1f} मीटर खाली), विहिरी व बोअरवेलचा उपसा मर्यादित.")
    elif groundwater >= 20.0:
        reasons_en.append(f"Moderate aquifer depth ({groundwater:.1f}m below ground), requiring controlled pumping.")
        reasons_mr.append(f"भूजल पातळी मध्यम खोल ({groundwater:.1f} मीटर), पाण्याचा अतिउपसा टाळावा.")
    else:
        reasons_en.append(f"Healthy groundwater recharge level ({groundwater:.1f}m below ground).")
        reasons_mr.append(f"भूजल पातळी समाधानकारक स्थितीमध्ये ({groundwater:.1f} मीटर).")

    # 4. Temperature Stress Evaluation
    if temp >= 35.0:
        reasons_en.append(f"High atmospheric thermal stress ({temp:.1f}°C) accelerating evapotranspiration loss.")
        reasons_mr.append(f"जास्त तापमान ({temp:.1f}°C) यामुळे बाष्पीभवनाचा वेग वाढून पाण्याचा ताण तीव्र होतो.")
    elif temp >= 33.0:
        reasons_en.append(f"Warm daytime temperature ({temp:.1f}°C) causing moderate moisture loss.")
        reasons_mr.append(f"उष्ण तापमान ({temp:.1f}°C), ओलावा टिकवून ठेवण्यासाठी आच्छादन गरजेचे.")
    else:
        reasons_en.append(f"Moderate temperature profile ({temp:.1f}°C), normal moisture evaporation.")
        reasons_mr.append(f"अनुकूल तापमान ({temp:.1f}°C), बाष्पीभवन मर्यादित.")

    # 5. Composite Risk Summary
    if prediction == "High":
        reasons_en.insert(0, "Multi-variable stress: Composite agro-meteorological indicators confirm severe water scarcity risk.")
        reasons_mr.insert(0, "बहु-घटक ताण: पावसाची तूट, कमी ओलावा आणि खोल भूजल यामुळे तीव्र दुष्काळी परिस्थिती निश्चित होते.")
    elif prediction == "Medium":
        reasons_en.insert(0, "Vulnerability warning: Intermediate stress factors detected; risk may escalate if dry spell prolongs.")
        reasons_mr.insert(0, "सावधानतेचा इशारा: मध्यम ताण दिसून येत आहे; पावसाचा खंड वाढल्यास जोखीम तीव्र होऊ शकते.")
    else:
        reasons_en.insert(0, "Stable condition: Adequate moisture buffer and low rainfall deficit present.")
        reasons_mr.insert(0, "समाधानकारक स्थिती: पुरेसा ओलावा आणि पाऊस समाधानकारक असल्याने जोखीम कमी आहे.")

    return reasons_mr if lang == "mr" else reasons_en


def generate_farmer_advisory(prediction, deficit, soil, groundwater, lang="en"):
    """
    Generates actionable, category-specific agronomic advisories based on risk.
    """
    if prediction == "High":
        advisories_en = {
            "Irrigation Management": [
                "Strictly avoid flood irrigation; operate micro-drip or sprinkler systems during early morning or evening hours only.",
                "Adopt alternate furrow irrigation to cut irrigation water consumption by 35-40%.",
                "Prioritize life-saving protective irrigation strictly at critical crop growth stages (flowering & pod/grain filling)."
            ],
            "Crop & Varietal Strategy": [
                "Avoid high-water consuming cash crops (such as Sugarcane or summer paddy).",
                "Promote drought-hardy, short-duration crops: Pearl Millet (Bajra - GHB 538), Sorghum (Maldandi/Phule Suchitra), Pigeonpea (BDN 711/716), Horsegram (Kulthi).",
                "If primary standing crop suffers >50% failure, immediately prepare for contingency fodder crops (Fodder Sorghum, Maize, Cowpea)."
            ],
            "Soil Moisture Conservation": [
                "Apply organic mulching (crop residue, sugarcane trash, dry grass, or 25-micron silver-black plastic mulch) to arrest soil evaporation.",
                "Implement shallow inter-culturing (hoeing) to create a dust mulch and break soil capillaries.",
                "Adopt Broad Bed Furrow (BBF) layout and contour bunding for in-situ rainwater harvesting."
            ],
            "Government Relief & Support": [
                "Register crop status under the Maharashtra 12-point Drought Relief Package (GR declared for affected talukas).",
                "Avail input subsidies, loan recovery standstill, and student exam fee waiver as per district collector orders.",
                "Contact nearest Tahsildar / Taluka Krishi Adhikari for livestock fodder depot (चारा डेपो) allocations."
            ]
        }

        advisories_mr = {
            "पाणी व सिंचन व्यवस्थापन": [
                "मोकाट/पाट पाणी देणे पूर्णपणे थांबवा; फक्त ठिबक किंवा तुषार सिंचनाचा वापर सकाळी किंवा सायंकाळी करा.",
                "एक आड एक सरी पद्धत (Alternate Furrow) वापरून ३५ ते ४०% पाण्याची बचत करा.",
                "पिकांच्या संवेदनशील अवस्थेतच (फुलधारणा व दाणे भरण्याची वेळ) केवळ संजीवनी/संरक्षित पाणी द्या."
            ],
            "पीक व बियाणे निवड": [
                "जास्त पाणी लागणारी पिके (उदा. ऊस किंवा उन्हाळी भात) घेणे टाळा.",
                "कमी कालावधीत येणारी व दुष्काळ सहन करणारी पिके निवडा: बाजरी (GHB 538), रब्बी ज्वारी (मालदांडी/फुले सुचित्रा), तूर (BDN 711), हुलगा (कुळीथ).",
                "उभ्या पिकाचे नुकसान ५०% पेक्षा जास्त असल्यास तातडीने जनावरांच्या चाऱ्यासाठी चारा ज्वारी किंवा चवळीची पेरणी करा."
            ],
            "मातीतील ओलावा टिकवणे": [
                "जमिनीतील ओलावा टिकवण्यासाठी काडीकचरा, पाचट किंवा भुशाचे आच्छादन (Mulching) करा.",
                "वरच्या थरातील जमिनीची खुरपणी/डवरणी करून 'धूळ आच्छादन' (Dust Mulch) तयार करा ज्यामुळे बाष्पीभवन थांबेल.",
                "रुंद वरंबा-सरी पद्धत (BBF) किंवा समपातळी बांध घालून उपलब्ध पाण्याचे मूलस्थानी संवर्धन करा."
            ],
            "शासकीय मदत व संपर्क": [
                "महाराष्ट्र शासनाच्या १२-कलमी दुष्काळ मदत पॅकेज अंतर्गत आपल्या तालुक्याची नोंद तपासा.",
                "पीक नुकसान भरपाई, कर्जवसुलीस स्थगिती आणि परीक्षा शुल्क माफीसाठी कृषी सहाय्यकांशी संपर्क साधा.",
                "जनावरांच्या पाण्यासाठी व चारा डेपोसाठी (Fodder Depot) ग्रामपंचायत अथवा तालुका कृषी अधिकाऱ्यांशी संपर्क साधा."
            ]
        }
    elif prediction == "Medium":
        advisories_en = {
            "Irrigation Management": [
                "Maintain scheduled irrigation cycles; avoid over-watering that depletes limited borewell storage.",
                "Inspect drip emitters and clean filtration units to maintain optimal water delivery efficiency."
            ],
            "Crop & Varietal Strategy": [
                "Spray 2% Urea or Potassium Nitrate (13:0:45) at flowering stage to enhance physiological drought tolerance.",
                "Thin overcrowded plants (remove weak seedlings) to reduce competition for soil moisture."
            ],
            "Soil Moisture Conservation": [
                "Apply light organic mulching between rows before temperatures peak.",
                "Create compartment bunds in farm fields to capture any intermittent shower."
            ],
            "Government Relief & Support": [
                "Ensure active enrollment under Pradhan Mantri Fasal Bima Yojana (PMFBY).",
                "Monitor district agricultural weather bulletins issued by Vasantrao Naik Marathwada Krishi Vidyapeeth (VNMKV), Parbhani."
            ]
        }

        advisories_mr = {
            "पाणी व सिंचन व्यवस्थापन": [
                "ठिबक सिंचनाचे नियोजन करा; विहिरीतील पाण्याची पातळी लक्षात घेऊन पाण्याचा योग्य वापर करा.",
                "फिल्टर आणि ड्रीपर्स नियमित स्वच्छ ठेवा जेणेकरून पाण्याचा अपव्यय होणार नाही."
            ],
            "पीक व बियाणे निवड": [
                "पिकाची ताण सहन करण्याची क्षमता वाढवण्यासाठी २% युरिया किंवा १३:०:४५ (पोटॅशियम नायट्रेट) ची फवारणी करा.",
                "रोपांची विरळणी करून एकरी झाडांची योग्य संख्या ठेवा जेणेकरून ओलाव्यासाठी स्पर्धा होणार नाही."
            ],
            "मातीतील ओलावा टिकवणे": [
                "पिकांच्या ओळींमध्ये सेंद्रिय आच्छादन करा जेणेकरून दुपारच्या उन्हात ओलावा उडून जाणार नाही.",
                "शेतजमिनीत समपातळीवर बांध किंवा खड्डे तयार ठेवा जेणेकरून येणारा पाऊस जमिनीत जिरवता येईल."
            ],
            "शासकीय मदत व संपर्क": [
                "पंतप्रधान पीक विमा योजना (PMFBY) मध्ये वेळेत अर्ज भरल्याची खात्री करा.",
                "वसंतराव नाईक मराठवाडा कृषी विद्यापीठ (परभणी) यांच्या कृषी हवामान सल्ल्याचे नियमित पालन करा."
            ]
        }
    else:  # Low Risk
        advisories_en = {
            "Irrigation Management": [
                "Soil moisture and groundwater levels are adequate. Maintain regular scientific crop watering intervals.",
                "Utilize farm ponds (Shet-tale) to store excess surplus runoff for upcoming dry spells."
            ],
            "Crop & Varietal Strategy": [
                "Ideal conditions for standard Kharif/Rabi crops (Soybean, Cotton, Gram, Wheat).",
                "Follow integrated nutrient and pest management schedules recommended for Marathwada."
            ],
            "Soil Moisture Conservation": [
                "Practice regular weed control to ensure all stored soil moisture benefits crops.",
                "Deep summer ploughing was beneficial; maintain soil aeration through light intercultural operations."
            ],
            "Government Relief & Support": [
                "Continue soil health card testing and solar agri-pump scheme participation."
            ]
        }

        advisories_mr = {
            "पाणी व सिंचन व्यवस्थापन": [
                "सध्या पाण्याचा साठा व ओलावा चांगला आहे. पिकांच्या आवश्यकतेनुसार नियमित पाणी द्या.",
                "शेततळे किंवा बंधाऱ्यांमध्ये अतिरिक्त पाणी साठवून ठेवा जेणेकरून पुढील हंगामात उपयोग होईल."
            ],
            "पीक व बियाणे निवड": [
                "नियमित खरीप व रब्बी पिकांसाठी (सोयाबीन, कापूस, हरभरा, गहू) वातावरण अनुकूल आहे.",
                "संतुलित खत व्यवस्थापन आणि एकात्मिक कीड व्यवस्थापन पद्धती अवलंबा."
            ],
            "मातीतील ओलावा टिकवणे": [
                "तणांचा वेळेवर बंदोबस्त करा, जेणेकरून तण पिकांचे पाणी शोषून घेणार नाहीत.",
                "जमिनीची सुपीकता टिकवण्यासाठी शेणखत व सेंद्रिय खतांचा नियमित वापर करा."
            ],
            "शासकीय मदत व संपर्क": [
                "माती आरोग्य पत्रिका (Soil Health Card) तपासून खतांचे योग्य प्रमाण ठरवा."
            ]
        }

    return advisories_mr if lang == "mr" else advisories_en


def answer_farmer_chatbot_query(query_text, current_risk="High", lang="auto"):
    """
    Intelligent bilingual query responder for farmer questions.
    Uses semantic keyword heuristics to emulate an expert agricultural chatbot.
    """
    query = query_text.lower().strip()

    # Detect language if auto
    is_marathi = any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in query_text)
    if lang == "mr" or (lang == "auto" and is_marathi):
        # Marathi responses
        if any(w in query for w in ["पाणी", "सिंचन", "ठिबक", "तुषार", "कमी पाणी", "water"]):
            return (
                "💧 **पाणी व्यवस्थापन सल्ला:**\n"
                "१. मोकाट पाणी देणे टाळा आणि फक्त **ठिबक सिंचन (Drip)** किंवा **तुषार सिंचन (Sprinkler)** वापरा.\n"
                "२. बाष्पीभवन टाळण्यासाठी पाणी सकाळी ६ ते ९ किंवा संध्याकाळी ५ नंतरच द्या.\n"
                "३. एकाड एक सरी पद्धत (Alternate furrow) वापरल्यास ३५-४०% पाण्याची थेट बचत होते.\n"
                "४. पिकांना फक्त फुलोरा आणि दाणे भरण्याच्या नाजूक टप्प्यावरच संरक्षित पाणी द्या."
            )
        elif any(w in query for w in ["पीक", "बियाणे", "कोणते पीक", "पेरणी", "crop", "seed"]):
            return (
                "🌾 **दुष्काळ सहन करणारी शिफारशीत पिके (मराठवाडा):**\n"
                "१. **बाजरी:** आय.सी.टी.पी. ८२०३, जी.एच.बी. ५३८ (कमी पाण्यात ६५-७० दिवसांत येते).\n"
                "२. **रब्बी ज्वारी:** मालदांडी (M-35-1), फुले सुचित्रा, फुले अनुराधा.\n"
                "३. **तूर:** बी.डी.एन. ७११ किंवा ७१६ (मराठवाडा विभागासाठी सर्वोत्तम).\n"
                "४. **कडधान्ये:** हुलगा (कुळीथ), मटकी, चवळी - हे जमिनीचा ओलावा कमी असतानाही तग धरतात.\n"
                "५. ऊस, केळी किंवा उन्हाळी भात यांसारखी जास्त पाणी खाणारी पिके सध्या पूर्णपणे टाळा."
            )
        elif any(w in query for w in ["ओलावा", "मल्चिंग", "आच्छादन", "moisture", "mulch"]):
            return (
                "🌱 **जमिनीतील ओलावा टिकवण्याचे उपाय:**\n"
                "१. **सेंद्रिय आच्छादन (Mulching):** उसाचे पाचट, सोयाबीनचे काड किंवा सुका पालापाचोळा ५ ते ७ सेंमी जाडीने पसरवा.\n"
                "२. **धूळ आच्छादन (Dust Mulching):** खुरपणी किंवा हलकी डवरणी करून जमिनीचा वरचा थर भुसभुशीत ठेवा.\n"
                "३. **बीबीएफ तंत्रज्ञान (BBF):** रुंद वरंबा सरी पद्धतीमुळे पावसाचे पाणी शेतातच मुरते आणि ओलावा टिकून राहतो.\n"
                "४. पोटॅशियम नायट्रेट (१३:०:४५) ची १-२% फवारणी केल्याने झाडांमधील पानांमधून होणारे बाष्पीभवन कमी होते."
            )
        elif any(w in query for w in ["मदत", "शासकीय", "अनुदान", "नुकसान", "पॅकेज", "योजना", "relief", "scheme"]):
            return (
                "🏛️ **शासकीय दुष्काळ मदत व योजना (महाराष्ट्र २०२६):**\n"
                "१. राज्य शासनाने मराठवाड्यातील ७४ तालुक्यांसह दुष्काळ जाहीर केला असून **१२-कलमी मदत पॅकेज** लागू आहे.\n"
                "२. **प्रमुख लाभ:** कृषी कर्जाच्या वसुलीस स्थगिती, वीज बिलात सवलत, शालेय/महाविद्यालयीन परीक्षा फी माफी.\n"
                "३. **पीक विमा:** पीएमएफबीवाय (PMFBY) अंतर्गत स्थानिक आपत्ती व दुष्काळी नुकसानीची पूर्वसूचना ७२ तासांच्या आत ॲपवर नोंदवा.\n"
                "४. **संपर्क:** अधिक मदतीसाठी आपल्या गावातील कृषी सहाय्यक, तलाठी किंवा तालुका कृषी कार्यालयाशी तात्काळ संपर्क साधा."
            )
        elif any(w in query for w in ["चारा", "जनावरे", "गाई", "म्हशी", "fodder", "cattle"]):
            return (
                "🐄 **जनावरांचे व चाऱ्याचे नियोजन:**\n"
                "१. दुष्काळग्रस्त भागात शासनातर्फे 'चारा डेपो' व छावण्यांचे नियोजन सुरू आहे.\n"
                "२. तातडीने हिरवा चारा मिळवण्यासाठी कमी पाण्यात येणाऱ्या 'आफ्रिकन टॉल' मका किंवा चारा ज्वारीची पेरणी करावी.\n"
                "३. उपलब्ध सुक्या चाऱ्यावर युरिया प्रक्रिया करून त्याची पौष्टिकता वाढवता येते.\n"
                "४. जनावरांच्या पिण्याच्या पाण्यासाठी ग्रामपंचायतीच्या टँकर व्यवस्थेची मदत घ्या."
            )
        else:
            return (
                "🤖 **शेतकरी मदतनीस:**\n"
                "मी मराठवाडा दुष्काळ निवारण कॉग्निटिव्ह सहाय्यक आहे.\n"
                "तुम्ही मला खालील विषयांवर विचारू शकता:\n"
                "• 'कमी पाण्यात सिंचन कसे करावे?'\n"
                "• 'दुष्काळात कोणती पिके घ्यावीत?'\n"
                "• 'जमिनीतील ओलावा कसा टिकवायचा?'\n"
                "• 'शासकीय दुष्काळ मदत व विमा योजना काय आहेत?'\n"
                "• 'जनावरांच्या चाऱ्याचे नियोजन कसे करावे?'"
            )
    else:
        # English responses
        if any(w in query for w in ["water", "irrigation", "drip", "sprinkler", "save water"]):
            return (
                "💧 **Irrigation Management Advice:**\n"
                "1. Discontinue surface flood irrigation immediately; utilize precision micro-drip or sprinkler irrigation.\n"
                "2. Schedule irrigation between 6:00 AM - 9:00 AM or after 5:30 PM to minimize evaporative loss.\n"
                "3. Use alternate furrow irrigation to conserve 35-40% of irrigation water.\n"
                "4. Restrict irrigation strictly to critical phenological stages (flowering and seed filling)."
            )
        elif any(w in query for w in ["crop", "seed", "variety", "sow", "plant"]):
            return (
                "🌾 **Drought-Tolerant Crops for Marathwada:**\n"
                "1. **Pearl Millet (Bajra):** ICTP-8203, GHB-538 (matures within 65-72 days on minimal moisture).\n"
                "2. **Rabi Sorghum (Jowar):** Maldandi (M-35-1), Phule Suchitra, Phule Anuradha.\n"
                "3. **Pigeonpea (Tur):** BDN-711 or BDN-716 (highly recommended for Marathwada clay soils).\n"
                "4. **Pulses:** Horsegram (Kulthi), Moth bean - resilient under acute moisture stress.\n"
                "5. Avoid water-intensive perennials like Sugarcane or summer paddy during declared drought conditions."
            )
        elif any(w in query for w in ["soil", "moisture", "mulch", "conserve"]):
            return (
                "🌱 **Soil Moisture Conservation Techniques:**\n"
                "1. **Organic Mulching:** Apply a 5-7 cm layer of crop stalks, soybean straw, or dry biomass.\n"
                "2. **Dust Mulching:** Conduct shallow hoeing to break capillary pores and trap subsurface moisture.\n"
                "3. **Broad Bed Furrow (BBF):** Facilitates in-situ percolation and reduces surface runoff.\n"
                "4. Foliar spray of 1-2% Potassium Nitrate (13:0:45) helps regulate stomatal closure and reduce transpiration."
            )
        elif any(w in query for w in ["scheme", "government", "subsidy", "relief", "package", "insurance"]):
            return (
                "🏛️ **Government Relief & Safety Schemes (Maharashtra 2026):**\n"
                "1. 74 talukas in Marathwada have been declared under drought with a **12-Point Relief Package**.\n"
                "2. **Key Entitlements:** Stay on agricultural loan recovery, electricity duty concessions, and student exam fee waivers.\n"
                "3. **PMFBY Insurance:** Intimate localized crop loss within 72 hours via the Crop Insurance App.\n"
                "4. **Action:** Contact your local Gram Sevak, Talathi, or Taluka Agriculture Officer (TAO) for enrollment."
            )
        elif any(w in query for w in ["fodder", "cattle", "livestock", "cow", "buffalo"]):
            return (
                "🐄 **Livestock & Fodder Management:**\n"
                "1. District administrations are establishing dedicated fodder depots (चारा डेपो) in drought-affected talukas.\n"
                "2. Sow short-duration green fodder such as African Tall Maize or Fodder Sorghum.\n"
                "3. Treat dry straw/fodder with 2% urea solution to enhance nutritional digestibility.\n"
                "4. Register tanker water requirements for livestock drinking water with the local Gram Panchayat."
            )
        else:
            return (
                "🤖 **Farmer Cognitive Advisory Assistant:**\n"
                "I am your Marathwada Drought Risk AI Assistant.\n"
                "You can ask me questions such as:\n"
                "• 'How to manage water in severe drought?'\n"
                "• 'Which drought-resistant crops are best for Marathwada?'\n"
                "• 'How can I conserve soil moisture using mulching?'\n"
                "• 'What government relief packages and insurance schemes are available?'\n"
                "• 'How to manage fodder and water for livestock?'"
            )
