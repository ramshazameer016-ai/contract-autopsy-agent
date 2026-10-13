"""
Multilingual translation engine and dictionary for Contract Autopsy Agent.
Provides deterministic, verifiable offline translations for:
- English
- Hindi (Devanagari script)
- Telugu (Telugu script)
- Tamil (Tamil script)
Strictly adheres to preserving verbatim contract quotes, dates, numbers, and facts.
"""

from __future__ import annotations

import re
from typing import Any

from src.ca.llm.validator import PlainSummary, SummaryItem

SUPPORTED_LANGUAGES: dict[str, str] = {
    "english": "English",
    "hindi": "Hindi",
    "telugu": "Telugu",
    "tamil": "Tamil",
}

DISCLAIMERS: dict[str, str] = {
    "English": (
        "This summary is informational and is not legal advice. "
        "Check all points against the original contract."
    ),
    "Hindi": (
        "यह सारांश केवल जानकारी के लिए है, कानूनी सलाह नहीं। "
        "सभी बिंदुओं को मूल अनुबंध से जाँचें।"
    ),
    "Telugu": (
        "ఈ సారాంశం సమాచారం కోసం మాత్రమే; ఇది న్యాయ సలహా కాదు. "
        "అన్ని అంశాలను అసలు ఒప్పందంతో తనిఖీ చేయండి."
    ),
    "Tamil": (
        "இந்தச் சுருக்கம் தகவலுக்காக மட்டுமே; இது சட்ட ஆலோசனை அல்ல. "
        "அனைத்து அம்சங்களையும் அசல் ஒப்பந்தத்துடன் சரிபார்க்கவும்."
    ),
}

CATEGORY_TRANSLATIONS: dict[str, dict[str, str]] = {
    "payment": {
        "English": "Payment terms",
        "Hindi": "भुगतान की शर्तें",
        "Telugu": "చెల్లింపు నిబంధనలు",
        "Tamil": "பணம் செலுத்தும் விதிமுறைகள்",
    },
    "ip": {
        "English": "Intellectual property",
        "Hindi": "बौद्धिक संपदा",
        "Telugu": "మేధो సంపత్తి",
        "Tamil": "அறிவுசார் சொத்துரிமை",
    },
    "termination": {
        "English": "Ending the contract",
        "Hindi": "अनुबंध समाप्ति",
        "Telugu": "ఒప్పందం రద్దు",
        "Tamil": "ஒப்பந்தத்தை முடித்தல்",
    },
    "scope": {
        "English": "Work and deliverables",
        "Hindi": "कार्य और डिलिवरेबल्स",
        "Telugu": "పని మరియు డెలివరీలు",
        "Tamil": "வேலை மற்றும் விநியோகங்கள்",
    },
    "liability": {
        "English": "Responsibility and liability",
        "Hindi": "दायित्व और जवाबदेही",
        "Telugu": "బాధ్యత మరియు పరిమితి",
        "Tamil": "பொறுப்பு மற்றும் வரையறை",
    },
    "confidentiality": {
        "English": "Confidentiality",
        "Hindi": "गोपनीयता",
        "Telugu": "రహస్యత",
        "Tamil": "ரகசியத்தன்மை",
    },
    "extracted clause": {
        "English": "Extracted clause",
        "Hindi": "निकाला गया खंड",
        "Telugu": "సేకరించిన నిబంధన",
        "Tamil": "பிரித்தெடுக்கப்பட்ட பிரிவு",
    },
    "contract text": {
        "English": "Contract text",
        "Hindi": "अनुबंध पाठ",
        "Telugu": "ఒప్పంద పాఠ్యం",
        "Tamil": "ஒப்பந்த உரை",
    },
}

STATUS_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Stated": {
        "English": "Stated in contract",
        "Hindi": "अनुबंध में उल्लिखित",
        "Telugu": "ఒప్పందంలో పేర్కొనబడింది",
        "Tamil": "ஒப்பந்தத்தில் குறிப்பிடப்பட்டுள்ளது",
    },
    "Missing": {
        "English": "Missing from contract",
        "Hindi": "अनुबंध में अनुपस्थित",
        "Telugu": "ఒప్పందంలో లేదు",
        "Tamil": "ஒப்பந்தத்தில் இல்லை",
    },
    "Unclear": {
        "English": "Unclear terms",
        "Hindi": "अस्पष्ट शर्तें",
        "Telugu": "అస్పష్టమైన నిబంధనలు",
        "Tamil": "தெளிவற்ற விதிமுறைகள்",
    },
    "Inferred": {
        "English": "Inferred",
        "Hindi": "अनुमानित",
        "Telugu": "ఊహించినది",
        "Tamil": "ஊகிக்கப்பட்டது",
    },
}

# Exact sentence and phrase dictionary for deterministic rule engine outputs
EXACT_REASONS: dict[str, dict[str, str]] = {
    # Combined and single absence reasons
    "no payment terms found. they may be missing, or text extraction missed them.": {
        "Hindi": "कोई भुगतान शर्तें नहीं मिलीं। वे अनुबंध में गायब हो सकती हैं, या पाठ निष्कर्षण में छूट गई हैं।",
        "Telugu": "చెల్లింపు నిబంధనలు ఏవీ కనుగొనబడలేదు. అవి ఒప్పందంలో లేకపోవచ్చు లేదా సంగ్రహణలో తప్పిపోయి ఉండవచ్చు.",
        "Tamil": "பணம் செலுத்தும் விதிமுறைகள் எதுவும் காணப்படவில்லை. அவை விடுபட்டிருக்கலாம் அல்லது உரை பிரித்தெடுத்தலில் தவறியிருக்கலாம்.",
    },
    "no payment terms found.": {
        "Hindi": "कोई भुगतान शर्तें नहीं मिलीं।",
        "Telugu": "చెల్లింపు నిబంధనలు ఏవీ కనుగొనబడలేదు.",
        "Tamil": "பணம் செலுத்தும் விதிமுறைகள் எதுவும் காணப்படவில்லை.",
    },
    "they may be missing, or text extraction missed them.": {
        "Hindi": "वे अनुबंध में गायब हो सकती हैं, या पाठ निष्कर्षण में छूट गई हैं।",
        "Telugu": "అవి ఒప్పందంలో లేకపోవచ్చు లేదా సంగ్రహణలో తప్పిపోయి ఉండవచ్చు.",
        "Tamil": "அவை விடுபட்டிருக்கலாம் அல்லது உரை பிரித்தெடுத்தலில் தவறியிருக்கலாம்.",
    },
    "no ip ownership clause found. ownership may be unclear; ask the client to state it.": {
        "Hindi": "कोई बौद्धिक संपदा (IP) स्वामित्व खंड नहीं मिला। स्वामित्व अस्पष्ट हो सकता है; ग्राहक से इसे स्पष्ट करने को कहें।",
        "Telugu": "మేధో సంపత్తి (IP) యాజమాన్య నిబంధన కనుగొనబడలేదు. యాజమాన్యం అస్పష్టంగా ఉండవచ్చు; క్లయింట్‌ను స్పష్టం చేయమని కోరండి.",
        "Tamil": "அறிவுசார் சொத்து (IP) உரிமைப் பிரிவு எதுவும் காணப்படவில்லை. உரிமை தெளிவாக இல்லாமல் இருக்கலாம்; வாடிக்கையாளரிடம் தெளிவுபடுத்துமாறு கேளுங்கள்.",
    },
    "no ip ownership clause found.": {
        "Hindi": "कोई बौद्धिक संपदा (IP) स्वामित्व खंड नहीं मिला।",
        "Telugu": "మేధో సంపత్తి (IP) యాజమాన్య నిబంధన కనుగొనబడలేదు.",
        "Tamil": "அறிவுசார் சொத்து (IP) உரிமைப் பிரிவு எதுவும் காணப்படவில்லை.",
    },
    "ownership may be unclear; ask the client to state it.": {
        "Hindi": "स्वामित्व अस्पष्ट हो सकता है; ग्राहक से इसे स्पष्ट करने को कहें।",
        "Telugu": "యాజమాన్యం అస్పష్టంగా ఉండవచ్చు; క్లయింట్‌ను స్పష్టం చేయమని కోరండి.",
        "Tamil": "உரிமை தெளிவாக இல்லாமல் இருக்கலாம்; வாடிக்கையாளரிடம் தெளிவுபடுத்துமாறு கேளுங்கள்.",
    },
    "no termination clause found. exit rights and notice are undefined.": {
        "Hindi": "कोई अनुबंध समाप्ति खंड नहीं मिला। बाहर निकलने के अधिकार और नोटिस अवधि अपरिभाषित हैं।",
        "Telugu": "ఒప్పందం రద్దు నిబంధన కనుగొనబడలేదు. నిష్క్రమణ హక్కులు మరియు నోటీసు వ్యవధి నిర్వచించబడలేదు.",
        "Tamil": "ஒப்பந்த ரத்து பிரிவு எதுவும் காணப்படவில்லை. வெளியேறும் உரிமைகள் மற்றும் நோட்டீஸ் காலம் வரையறுக்கப்படவில்லை.",
    },
    "no termination clause found.": {
        "Hindi": "कोई अनुबंध समाप्ति खंड नहीं मिला।",
        "Telugu": "ఒప్పందం రద్దు నిబంధన కనుగొనబడలేదు.",
        "Tamil": "ஒப்பந்த ரத்து பிரிவு எதுவும் காணப்படவில்லை.",
    },
    "exit rights and notice are undefined.": {
        "Hindi": "बाहर निकलने के अधिकार और नोटिस अवधि अपरिभाषित हैं।",
        "Telugu": "నిష్క్రమణ హక్కులు మరియు నోటీసు వ్యవధి నిర్వచించబడలేదు.",
        "Tamil": "வெளியேறும் உரிமைகள் மற்றும் நோட்டீஸ் காலம் வரையறுக்கப்படவில்லை.",
    },
    "no scope or deliverables clause found. work boundaries are undefined.": {
        "Hindi": "कोई कार्यक्षेत्र या डिलिवरेबल्स खंड नहीं मिला। कार्य की सीमाएं अपरिभाषित हैं।",
        "Telugu": "పని పరిధి లేదా డెలివరీల నిబంధన కనుగొనబడలేదు. పని పరిమితులు నిర్వచించబడలేదు.",
        "Tamil": "வேலை வரம்பு அல்லது விநியோக பிரிவு எதுவும் காணப்படவில்லை. வேலைக்கான எல்லைகள் வரையறுக்கப்படவில்லை.",
    },
    "no scope or deliverables clause found.": {
        "Hindi": "कोई कार्यक्षेत्र या डिलिवरेबल्स खंड नहीं मिला।",
        "Telugu": "పని పరిధి లేదా డెలివరీల నిబంధన కనుగొనబడలేదు.",
        "Tamil": "வேலை வரம்பு அல்லது விநியோக பிரிவு எதுவும் காணப்படவில்லை.",
    },
    "work boundaries are undefined.": {
        "Hindi": "कार्य की सीमाएं अपरिभाषित हैं।",
        "Telugu": "పని పరిమితులు నిర్వచించబడలేదు.",
        "Tamil": "வேலைக்கான எல்லைகள் வரையறுக்கப்படவில்லை.",
    },
    "no limitation of liability found. you may be exposed to unlimited damages.": {
        "Hindi": "देयता की कोई सीमा नहीं मिली। आप पर असीमित क्षतिपूर्ति का जोखिम हो सकता है।",
        "Telugu": "బాధ్యత పరిమితి నిబంధన కనుగొనబడలేదు. మీరు అపరిమిత నష్టపరిహార బాధ్యతను ఎదుర్కొనే ప్రమాదం ఉంది.",
        "Tamil": "பொறுப்பு வரம்பு எதுவும் காணப்படவில்லை. நீங்கள் வரம்பற்ற இழப்பீட்டுப் பொறுப்புக்கு ஆளாக நேரிடலாம்.",
    },
    "no limitation of liability found.": {
        "Hindi": "देयता की कोई सीमा नहीं मिली।",
        "Telugu": "బాధ్యత పరిమితి నిబంధన కనుగొనబడలేదు.",
        "Tamil": "பொறுப்பு வரம்பு எதுவும் காணப்படவில்லை.",
    },
    "you may be exposed to unlimited damages.": {
        "Hindi": "आप पर असीमित क्षतिपूर्ति का जोखिम हो सकता है।",
        "Telugu": "మీరు అపరిమిత నష్టపరిహార బాధ్యతను ఎదుర్కొనే ప్రమాదం ఉంది.",
        "Tamil": "நீங்கள் வரம்பற்ற இழப்பீட்டுப் பொறுப்புக்கு ஆளாக நேரிடலாம்.",
    },
    "no confidentiality terms found. proprietary information may be unprotected.": {
        "Hindi": "कोई गोपनीयता शर्तें नहीं मिलीं। आपकी मालिकाना जानकारी असुरक्षित हो सकती है।",
        "Telugu": "రహస్యత నిబంధనలు ఏవీ కనుగొనబడలేదు. యాజమాన్య సమాచారం అసురక్షితంగా ఉండవచ్చు.",
        "Tamil": "ரகசியத்தன்மை விதிமுறைகள் எதுவும் காணப்படவில்லை. தனியுரிமத் தகவல் பாதுகாக்கப்படாமல் போகலாம்.",
    },
    "no confidentiality terms found.": {
        "Hindi": "कोई गोपनीयता शर्तें नहीं मिलीं।",
        "Telugu": "రహస్యత నిబంధనలు ఏవీ కనుగొనబడలేదు.",
        "Tamil": "ரகசியத்தன்மை விதிமுறைகள் எதுவும் காணப்படவில்லை.",
    },
    "proprietary information may be unprotected.": {
        "Hindi": "आपकी मालिकाना जानकारी असुरक्षित हो सकती है।",
        "Telugu": "యాజమాన్య సమాచారం అసురక్షితంగా ఉండవచ్చు.",
        "Tamil": "தனியுரிமத் தகவல் பாதுகாக்கப்படாமல் போகலாம்.",
    },

    # Payment rules
    "payment depends on the client being paid by someone else (pay-when-paid).": {
        "Hindi": "भुगतान इस बात पर निर्भर करता है कि ग्राहक को किसी अन्य से भुगतान मिले (pay-when-paid)।",
        "Telugu": "చెల్లింపు అనేది క్లయింట్‌కు వేరొకరి నుండి డబ్బు అందడంపై ఆధారపడి ఉంటుంది (pay-when-paid).",
        "Tamil": "வாடிக்கையாளருக்கு வேறொருவரிடமிருந்து பணம் கிடைப்பதைப் பொறுத்தே உங்களுக்குப் பணம் வழங்கப்படும் (pay-when-paid).",
    },
    "a deadline is set but no late-payment interest or remedy is stated.": {
        "Hindi": "समय-सीमा निर्धारित है लेकिन विलंब शुल्क या ब्याज का कोई उल्लेख नहीं है।",
        "Telugu": "గడువు నిర్ణయించబడింది కానీ ఆలస్య చెల్లింపు వడ్డీ లేదా పరిష్కారం పేర్కొనబడలేదు.",
        "Tamil": "காலக்கெடு நிர்ணயிக்கப்பட்டுள்ளது, ஆனால் தாமதக் கட்டணம் அல்லது வட்டி எதுவும் குறிப்பிடப்படவில்லை.",
    },
    "no clear payment deadline (for example 'reasonable time').": {
        "Hindi": "भुगतान की कोई स्पष्ट समय-सीमा नहीं है (उदा. 'उचित समय')।",
        "Telugu": "స్పష్టమైన చెల్లింపు గడువు లేదు (ఉదాహరణకు 'సహేతుకమైన సమయం').",
        "Tamil": "தெளிவான கட்டண காலக்கெடு இல்லை (உதாரணமாக 'நியாயமான நேரம்').",
    },
    "advance or milestone payments reduce your exposure.": {
        "Hindi": "अग्रिम या मील का पत्थर (माइलस्टोन) आधारित भुगतान आपके वित्तीय जोखिम को कम करता है।",
        "Telugu": "ముందస్తు లేదా మైలురాయి ఆధారిత చెల్లింపులు మీ రిస్క్‌ను తగ్గిస్తాయి.",
        "Tamil": "முன்பணம் அல்லது மைல்கல் அடிப்படையிலான கட்டணம் உங்கள் நிதி அபாயத்தைக் குறைக்கிறது.",
    },
    "a payment deadline is stated.": {
        "Hindi": "भुगतान की समय-सीमा बताई गई है।",
        "Telugu": "చెల్లింపు గడువు పేర్కొనబడింది.",
        "Tamil": "கட்டண காலக்கெடு குறிப்பிடப்பட்டுள்ளது.",
    },
    "payment terms need review.": {
        "Hindi": "भुगतान की शर्तों की समीक्षा आवश्यक है।",
        "Telugu": "చెల్లింపు నిబంధనలను సమీక్షించాలి.",
        "Tamil": "பணம் செலுத்தும் விதிமுறைகளை மதிப்பாய்வு செய்ய வேண்டும்.",
    },
    "payment timing is unclear.": {
        "Hindi": "भुगतान का समय अस्पष्ट है।",
        "Telugu": "చెల్లింపు సమయం అస్పష్టంగా ఉంది.",
        "Tamil": "பணம் செலுத்தும் நேரம் தெளிவாக இல்லை.",
    },
    "payment: stated": {
        "Hindi": "भुगतान: अनुबंध में उल्लिखित।",
        "Telugu": "చెల్లింపు: ఒప్పందంలో పేర్కొనబడింది.",
        "Tamil": "கட்டணம்: ஒப்பந்தத்தில் குறிப்பிடப்பட்டுள்ளது.",
    },

    # IP rules
    "all work is assigned with no carve-out for your pre-existing tools or material.": {
        "Hindi": "आपकी पूर्व-मौजूद सामग्री या टूल्स को अलग किए बिना सभी कार्य सौंप दिए गए हैं।",
        "Telugu": "మీ మునుపటి టూల్స్ లేదా మెటీరియల్‌కు ఎలాంటి మినహాయింపు లేకుండా మొత్తం పని కేటాయించబడింది.",
        "Tamil": "உங்கள் முன்பிருந்த கருவிகள் அல்லது பொருட்களுக்கு எந்த விலக்கும் இல்லாமல் அனைத்து பணிகளும் வாடிக்கையாளருக்கு மாற்றப்படுகின்றன.",
    },
    "ownership passes when work is created, possibly before you are paid.": {
        "Hindi": "कार्य बनते ही स्वामित्व स्थानांतरित हो जाता है, संभवतः भुगतान मिलने से पहले ही।",
        "Telugu": "పని సృష్టించబడిన వెంటనే యాజమాన్యం బదిలీ అవుతుంది, బహుశా మీకు చెల్లింపు అందకముందే.",
        "Tamil": "வேலை உருவாக்கப்பட்ட உடனேயே உரிமை மாறுகிறது, ஒருவேளை உங்களுக்கு பணம் கிடைப்பதற்கு முன்பே.",
    },
    "you are asked to waive moral rights.": {
        "Hindi": "आपसे अपने नैतिक अधिकारों को छोड़ने (वेव करने) के लिए कहा गया है।",
        "Telugu": "మీ నైతిక హక్కులను వదులుకోవాలని అడగబడ్డారు.",
        "Tamil": "உங்கள் தார்மீக உரிமைகளை விட்டுக்கொடுக்குமாறு கேட்கப்படுகிறீர்கள்.",
    },
    "you keep ownership and grant a licence instead of assigning it.": {
        "Hindi": "आप स्वामित्व अपने पास रखते हैं और सौंपने के बजाय केवल लाइसेंस प्रदान करते हैं।",
        "Telugu": "మీరు యాజమాన్యాన్ని మీ వద్దే ఉంచుకుని, బదిలీ చేయడానికి బదులుగా లైసెన్స్ ఇస్తారు.",
        "Tamil": "நீங்கள் உரிமையை உங்களிடமே வைத்துக்கொண்டு, அதற்குப் பதிலாக உரிமத்தை வழங்குகிறீர்கள்.",
    },
    "only final deliverables transfer, and only after full payment.": {
        "Hindi": "केवल अंतिम डिलिवरेबल्स स्थानांतरित होते हैं, और वह भी पूर्ण भुगतान के बाद ही।",
        "Telugu": "తుది డెలివరీలు మాత్రమే బదిలీ చేయబడతాయి, అదీ పూర్తి చెల్లింపు తర్వాత మాత్రమే.",
        "Tamil": "இறுதி விநியோகங்கள் மட்டுமே மாற்றப்படுகின்றன, அதுவும் முழு கட்டணம் செலுத்திய பின்னரே.",
    },

    # Termination rules
    "the client can end the contract at any time without notice and without paying for work done.": {
        "Hindi": "ग्राहक बिना किसी नोटिस और बिना किए गए काम का भुगतान किए कभी भी अनुबंध समाप्त कर सकता है।",
        "Telugu": "క్లయింట్ ఎటువంటి నోటీసు లేకుండా మరియు చేసిన పనికి చెల్లించకుండా ఎప్పుడైనా ఒప్పందాన్ని రద్దు చేయవచ్చు.",
        "Tamil": "வாடிக்கையாளர் எந்தவித முன்னறிவிப்பும் இன்றி, செய்த வேலைக்கு பணம் தராமலும் எப்போது வேண்டுமானாலும் ஒப்பந்தத்தை முடிக்கலாம்.",
    },
    "only the client may terminate for convenience; you have no matching right.": {
        "Hindi": "केवल ग्राहक को अपनी सुविधा अनुसार समाप्त करने का अधिकार है; आपके पास कोई समान अधिकार नहीं है।",
        "Telugu": "క్లయింట్‌కు మాత్రమే సౌలభ్యం కోసం రద్దు చేసుకునే హక్కు ఉంది; మీకు అటువంటి హక్కు లేదు.",
        "Tamil": "வாடிக்கையாளருக்கு மட்டுமே வசதிக்காக ஒப்பந்தத்தை முடிக்கும் உரிமை உள்ளது; உங்களுக்கு அத்தகைய உரிமை இல்லை.",
    },
    "notice period is under 7 days.": {
        "Hindi": "नोटिस अवधि 7 दिनों से कम है।",
        "Telugu": "నోటీసు వ్యవధి 7 రోజుల కంటే తక్కువగా ఉంది.",
        "Tamil": "நோட்டீஸ் காலம் 7 நாட்களுக்குக் குறைவாக உள்ளது.",
    },
    "at least 14 days notice and payment for work completed.": {
        "Hindi": "कम से कम 14 दिनों का नोटिस और पूरे किए गए कार्य का भुगतान।",
        "Telugu": "కనీసం 14 రోజుల నోటీసు మరియు పూర్తి చేసిన పనికి చెల్లింపు.",
        "Tamil": "குறைந்தது 14 நாட்கள் நோட்டீஸ் மற்றும் முடிக்கப்பட்ட வேலைக்கான கட்டணம்.",
    },

    # Scope rules
    "unlimited revisions with no extra pay.": {
        "Hindi": "बिना किसी अतिरिक्त भुगतान के असीमित संशोधन।",
        "Telugu": "అదనపు చెల్లింపు లేకుండా అపరిమిత సవరణలు.",
        "Tamil": "கூடுதல் கட்டணமின்றி வரம்பற்ற திருத்தங்கள்.",
    },
    "deliverables are not clearly defined.": {
        "Hindi": "डिलिवरेबल्स स्पष्ट रूप से परिभाषित नहीं हैं।",
        "Telugu": "డెలివరీలు స్పష్టంగా నిర్వచించబడలేదు.",
        "Tamil": "விநியோகங்கள் தெளிவாக வரையறுக்கப்படவில்லை.",
    },
    "open-ended wording such as 'and any other services required'.": {
        "Hindi": "अस्पष्ट एवं खुली शब्दावली जैसे कि 'और कोई अन्य आवश्यक सेवाएं'।",
        "Telugu": "'మరియు అవసరమైన ఇతర సేవలు' వంటి అస్పష్టమైన పదాలు.",
        "Tamil": "'மற்றும் தேவையான பிற சேவைகள்' போன்ற வரம்பற்ற சொற்கள்.",
    },
    "no process for pricing changes to the scope.": {
        "Hindi": "कार्यक्षेत्र में बदलाव के लिए मूल्य निर्धारण की कोई प्रक्रिया नहीं है।",
        "Telugu": "పని పరిధిలో మార్పులకు ధర నిర్ణయించే ప్రక్రియ లేదు.",
        "Tamil": "வேலை வரம்பு மாற்றங்களுக்கு கட்டணம் நிர்ணயிக்கும் செயல்முறை எதுவும் இல்லை.",
    },
    "defined deliverables, a revision limit and a change process.": {
        "Hindi": "परिभाषित डिलिवरेबल्स, एक संशोधन सीमा और परिवर्तन नियंत्रण प्रक्रिया।",
        "Telugu": "స్పష్టమైన డెలివరీలు, సవరణల పరిమితి మరియు మార్పుల ప్రక్రియ.",
        "Tamil": "வரையறுக்கப்பட்ட விநியோகங்கள், திருத்த வரம்பு மற்றும் மாற்றக் கட்டுப்பாட்டு செயல்முறை.",
    },

    # Liability rules
    "contractor liability is unlimited or open-ended.": {
        "Hindi": "फ्रीलांसर (ठेकेदार) का दायित्व असीमित या खुला है।",
        "Telugu": "కాంట్రాక్టర్ బాధ్యత అపరిమితంగా లేదా నిరవధికంగా ఉంది.",
        "Tamil": "ஒப்பந்ததாரரின் பொறுப்பு வரம்பற்றது அல்லது முடிவற்றது.",
    },
    "no aggregate liability cap is specified.": {
        "Hindi": "कुल देयता की कोई अधिकतम सीमा निर्दिष्ट नहीं है।",
        "Telugu": "మొత్తం బాధ్యత పరిమితి పేర్కొనబడలేదు.",
        "Tamil": "மொத்த பொறுப்பு வரம்பு எதுவும் குறிப்பிடப்படவில்லை.",
    },
    "liability is capped and indirect/consequential damages are excluded.": {
        "Hindi": "देयता सीमित है और अप्रत्यक्ष/परिणामी नुकसान को बाहर रखा गया है।",
        "Telugu": "బాధ్యత పరిమితం చేయబడింది మరియు పరోక్ష నష్టాలు మినహాయించబడ్డాయి.",
        "Tamil": "பொறுப்பு வரம்பிற்குட்பட்டது மற்றும் மறைமுக சேதங்கள் விலக்கப்பட்டுள்ளன.",
    },

    # Confidentiality rules
    "confidentiality obligations are unilateral rather than mutual.": {
        "Hindi": "गोपनीयता की शर्तें पारस्परिक होने के बजाय एकतरफा हैं।",
        "Telugu": "రహస్యత బాధ్యతలు పరస్పరం కాకుండా ఏకపక్షంగా ఉన్నాయి.",
        "Tamil": "ரகசியத்தன்மைக் கடமைகள் பரஸ்பரமானது அல்லாமல் ஒருதலைப்பட்சமாக உள்ளன.",
    },
    "confidentiality obligations are mutual.": {
        "Hindi": "गोपनीयता की शर्तें पारस्परिक हैं।",
        "Telugu": "రహస్యత బాధ్యతలు పరస్పరమైనవి.",
        "Tamil": "ரகசியத்தன்மைக் கடமைகள் பரஸ்பரமானவை.",
    },

    "no specific risk or favourable rule fired.": {
        "Hindi": "कोई विशिष्ट जोखिम या अनुकूल नियम लागू नहीं हुआ।",
        "Telugu": "నిర్దిష్ట రిస్క్ లేదా అనుకూల నిబంధన ఏదీ వర్తించలేదు.",
        "Tamil": "குறிப்பிட்ட அபாயம் அல்லது சாதகமான விதி எதுவும் பொருந்தவில்லை.",
    },
    "required facts are missing to confirm favorable terms.": {
        "Hindi": "अनुकूल शर्तों की पुष्टि के लिए आवश्यक तथ्य गायब हैं।",
        "Telugu": "అనుకూల నిబంధనలను నిర్ధారించడానికి అవసరమైన వాస్తవాలు లేవు.",
        "Tamil": "சாதகமான விதிமுறைகளை உறுதிப்படுத்த தேவையான உண்மைகள் இல்லை.",
    },

    # Previews & fallbacks
    "source-text preview only. verify the original contract before relying on this text.": {
        "Hindi": "केवल स्रोत-पाठ पूर्वावलोकन। इस पाठ पर भरोसा करने से पहले मूल अनुबंध की पुष्टि करें।",
        "Telugu": "మూల పాఠ్య ప్రివ్యూ మాత్రమే. ఈ పాఠ్యంపై ఆధారపడే ముందు అసలు ఒప్పందాన్ని ధృవీకరించండి.",
        "Tamil": "மூல உரை முன்னோட்டம் மட்டுமே. இந்த உரையை நம்புவதற்கு முன் அசல் ஒப்பந்தத்தை சரிபார்க்கவும்.",
    },
    "source-text preview only.": {
        "Hindi": "केवल स्रोत-पाठ पूर्वावलोकन।",
        "Telugu": "మూల పాఠ్య ప్రివ్యూ మాత్రమే.",
        "Tamil": "மூல உரை முன்னோட்டம் மட்டுமே.",
    },
    "verify the original contract before relying on this text.": {
        "Hindi": "इस पाठ पर भरोसा करने से पहले मूल अनुबंध की पुष्टि करें।",
        "Telugu": "ఈ పాఠ్యంపై ఆధారపడే ముందు అసలు ఒప్పందాన్ని ధృవీకరించండి.",
        "Tamil": "இந்த உரையை நம்புவதற்கு முன் அசல் ஒப்பந்தத்தை சரிபார்க்கவும்.",
    },
    "no readable contract text was available to generate a summary.": {
        "Hindi": "सारांश तैयार करने के लिए कोई पठनीय अनुबंध पाठ उपलब्ध नहीं था।",
        "Telugu": "సారాంశాన్ని రూపొందించడానికి చదవగలిగే ఒప్పంద పాఠ్యం ఏదీ అందుబాటులో లేదు.",
        "Tamil": "சுருக்கத்தை உருவாக்க படிக்கக்கூடிய ஒப்பந்த உரை எதுவும் கிடைக்கவில்லை.",
    },
}


def supported_languages() -> tuple[str, ...]:
    """Return the languages supported by the current summary pipeline."""
    return tuple(SUPPORTED_LANGUAGES.values())


def is_devanagari(text: str) -> bool:
    """Check if text contains Devanagari characters."""
    return bool(re.search(r"[\u0900-\u097F]", text))


def is_telugu(text: str) -> bool:
    """Check if text contains Telugu characters."""
    return bool(re.search(r"[\u0C00-\u0C7F]", text))


def is_tamil(text: str) -> bool:
    """Check if text contains Tamil characters."""
    return bool(re.search(r"[\u0B80-\u0BFF]", text))


def script_percentage(text: str, language: str) -> float:
    """Calculate the percentage of alphabetic characters in the target script."""
    lang = language.strip().casefold()
    if lang == "hindi":
        target_chars = len(re.findall(r"[\u0900-\u097F]", text))
    elif lang == "telugu":
        target_chars = len(re.findall(r"[\u0C00-\u0C7F]", text))
    elif lang == "tamil":
        target_chars = len(re.findall(r"[\u0B80-\u0BFF]", text))
    elif lang == "english":
        target_chars = len(re.findall(r"[a-zA-Z]", text))
    else:
        return 0.0

    all_letters = len(re.findall(r"[a-zA-Z\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF]", text))
    if all_letters == 0:
        return 1.0
    return target_chars / all_letters


def translate_category(category_raw: str, language: str) -> str:
    """Translate contract category into the requested language."""
    cat_key = category_raw.strip().casefold()
    lang_name = SUPPORTED_LANGUAGES.get(language.strip().casefold(), "English")

    if lang_name == "English":
        if cat_key in CATEGORY_TRANSLATIONS:
            return CATEGORY_TRANSLATIONS[cat_key]["English"]
        return category_raw.strip()

    if cat_key in CATEGORY_TRANSLATIONS:
        return CATEGORY_TRANSLATIONS[cat_key].get(lang_name, category_raw)

    return category_raw.strip()


def translate_status(status_raw: str, language: str) -> str:
    """Translate status label into the requested language."""
    status_key = status_raw.strip()
    lang_name = SUPPORTED_LANGUAGES.get(language.strip().casefold(), "English")

    if status_key in STATUS_TRANSLATIONS:
        return STATUS_TRANSLATIONS[status_key].get(lang_name, status_key)
    return status_key


def translate_disclaimer(language: str) -> str:
    """Return the legal disclaimer in the requested language."""
    lang_name = SUPPORTED_LANGUAGES.get(language.strip().casefold(), "English")
    return DISCLAIMERS.get(lang_name, DISCLAIMERS["English"])


def _translate_single_sentence(
    sentence: str,
    lang_name: str,
    allow_fallback_marker: bool = True,
) -> str | None:
    """Translate one sentence or rule explanation into target language."""
    clean = re.sub(r"\s+", " ", sentence).strip()
    clean_lower = clean.casefold()

    if not clean:
        return ""

    if lang_name == "English":
        return clean

    # 1. Exact match lookup
    if clean_lower in EXACT_REASONS:
        trans = EXACT_REASONS[clean_lower].get(lang_name)
        if trans:
            return trans

    # 2. Dynamic regex matches for parameterized facts
    # Pattern: Payment is due {value} days after invoice. You would finance the client for that long.
    m = re.match(
        r"^payment is due (\d+) days after invoice\.?\s*(?:you would finance the client for that long\.?)?$",
        clean_lower,
    )
    if m:
        days = m.group(1)
        if lang_name == "Hindi":
            return f"चालान के {days} दिनों के बाद भुगतान देय है। आप इतने लंबे समय तक ग्राहक को वित्तपोषित करेंगे।"
        elif lang_name == "Telugu":
            return f"ఇన్‌వాయిస్ వచ్చిన {days} రోజుల తర్వాత చెల్లింపు గడువు ఉంది. మీరు అంతకాలం క్లయింట్‌కు నిధులు సమకూర్చాల్సి ఉంటుంది."
        elif lang_name == "Tamil":
            return f"விலைப்பட்டியல் வந்த {days} நாட்களுக்குப் பிறகு பணம் செலுத்தப்பட வேண்டும். அவ்வளவு காலம் நீங்கள் வாடிக்கையாளருக்கு நிதியளிக்க வேண்டியிருக்கும்."

    # Pattern: Payment is due in {value} days, longer than the rubric target of {target} days.
    m = re.match(
        r"^payment is due in (\d+) days,? longer than the rubric target of (?:the rubric target of\s*)?(\d+|[a-zA-Z]+) days\.?$",
        clean_lower,
    )
    if m:
        days, target = m.group(1), m.group(2)
        if lang_name == "Hindi":
            return f"भुगतान {days} दिनों में देय है, जो लक्ष्य अवधि {target} दिनों से अधिक लंबा है।"
        elif lang_name == "Telugu":
            return f"చెల్లింపు {days} రోజుల్లో గడువు ఉంది, ఇది లక్ష్య కాలమైన {target} రోజుల కంటే ఎక్కువ."
        elif lang_name == "Tamil":
            return f"பணம் செலுத்தும் காலம் {days} நாட்கள் ஆகும், இது இலக்கு காலமான {target} நாட்களை விட அதிகம்."

    # Pattern: Clear amount and payment within {value} days.
    m = re.match(r"^clear amount and payment within (\d+) days\.?$", clean_lower)
    if m:
        days = m.group(1)
        if lang_name == "Hindi":
            return f"स्पष्ट राशि और {days} दिनों के भीतर भुगतान।"
        elif lang_name == "Telugu":
            return f"స్పష్టమైన మొత్తం మరియు {days} రోజుల్లో చెల్లింపు."
        elif lang_name == "Tamil":
            return f"தெளிவான தொகை மற்றும் {days} நாட்களுக்குள் கட்டணம்."

    # Pattern: Payment is due within {days} days.
    m = re.match(r"^payment is due within (\d+) days\.?$", clean_lower)
    if m:
        days = m.group(1)
        if lang_name == "Hindi":
            return f"भुगतान {days} दिनों के भीतर देय है।"
        elif lang_name == "Telugu":
            return f"చెల్లింపు {days} రోజుల్లోపు చెల్లించాలి."
        elif lang_name == "Tamil":
            return f"கட்டணம் {days} நாட்களுக்குள் செலுத்தப்பட வேண்டும்."

    # Pattern: Notice period is under {days} days.
    m = re.match(r"^notice period is under (\d+) days\.?$", clean_lower)
    if m:
        days = m.group(1)
        if lang_name == "Hindi":
            return f"नोटिस अवधि {days} दिनों से कम है।"
        elif lang_name == "Telugu":
            return f"నోటీసు వ్యవధి {days} రోజుల కంటే తక్కువగా ఉంది."
        elif lang_name == "Tamil":
            return f"நோட்டீஸ் காலம் {days} நாட்களுக்குக் குறைவாக உள்ளது."

    # Pattern: At least {days} days notice and payment for work completed.
    m = re.match(r"^at least (\d+) days notice and payment for work completed\.?$", clean_lower)
    if m:
        days = m.group(1)
        if lang_name == "Hindi":
            return f"कम से कम {days} दिनों का नोटिस और पूरे किए गए कार्य का भुगतान।"
        elif lang_name == "Telugu":
            return f"కనీసం {days} రోజుల నోటీసు మరియు పూర్తి చేసిన పనికి చెల్లింపు."
        elif lang_name == "Tamil":
            return f"குறைந்தது {days} நாட்கள் நோட்டீஸ் மற்றும் முடிக்கப்பட்ட வேலைக்கான கட்டணம்."

    # Pattern: The available analysis marks {category} as {status}.
    m = re.match(r"^the available analysis marks (.+?) as (.+?)\.?$", clean_lower)
    if m:
        cat_raw, stat_raw = m.group(1), m.group(2)
        cat_tr = translate_category(cat_raw, lang_name)
        stat_tr = translate_status(stat_raw.capitalize(), lang_name)
        if lang_name == "Hindi":
            return f"उपलब्ध विश्लेषण {cat_tr} को {stat_tr} के रूप में चिह्नित करता है।"
        elif lang_name == "Telugu":
            return f"అందుబాటులో ఉన్న విశ్లేషణ {cat_tr} ను {stat_tr} గా సూచిస్తుంది."
        elif lang_name == "Tamil":
            return f"கிடைக்கக்கூடிய பகுப்பாய்வு {cat_tr} ஐ {stat_tr} எனக் குறிக்கிறது."

    # Pattern: Cross-clause conflict / Clause conflicts with clause {id}
    m = re.search(r"conflict(?:s|ing)? with clause(?:s|\(s\))?:?\s*([a-zA-Z0-9\-_,\s]+)", clean_lower)
    if m:
        conflicts = m.group(1).strip()
        if lang_name == "Hindi":
            return f"खंडों के बीच परस्पर विरोधी शर्तें पाई गईं: {conflicts}।"
        elif lang_name == "Telugu":
            return f"నిబంధనల మధ్య పరస్పర విరుద్ధమైన షరతులు గుర్తించబడ్డాయి: {conflicts}."
        elif lang_name == "Tamil":
            return f"பிரிவுகளுக்கு இடையே முரண்பட்ட விதிமுறைகள் கண்டறியப்பட்டன: {conflicts}."

    # Pattern: Ambiguous wording found: '{quote}'.
    m = re.match(r"^ambiguous wording found:\s*['\"](.+?)['\"]\.?$", clean_lower)
    if m:
        amb_q = m.group(1)
        if lang_name == "Hindi":
            return f"अस्पष्ट शब्दावली मिली: '{amb_q}'।"
        elif lang_name == "Telugu":
            return f"అస్పష్టమైన పదాలు కనుగొనబడ్డాయి: '{amb_q}'."
        elif lang_name == "Tamil":
            return f"தெளிவற்ற சொற்கள் கண்டறியப்பட்டன: '{amb_q}'."

    if not allow_fallback_marker:
        return None

    # Honest fallback marker for untranslated text
    if lang_name == "Hindi":
        return f"[अनुवाद अनुपलब्ध]: {clean}"
    elif lang_name == "Telugu":
        return f"[అనువాదం అందుబాటులో లేదు]: {clean}"
    elif lang_name == "Tamil":
        return f"[மொழிபெயர்ப்பு கிடைக்கவில்லை]: {clean}"

    return clean


def translate_explanation(
    text: str,
    language: str,
    category: str = "",
    status: str = "",
) -> str:
    """
    Translate full explanation text into the target language.
    Handles compound sentences and multi-reason findings deterministically.
    """
    lang_name = SUPPORTED_LANGUAGES.get(language.strip().casefold(), "English")

    if lang_name == "English":
        return text.strip()

    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        if category and status:
            return _translate_single_sentence(
                f"The available analysis marks {category} as {status.lower()}.",
                lang_name,
                allow_fallback_marker=True,
            ) or ""
        return ""

    # 1. Try translating the entire string as a whole first
    whole_trans = _translate_single_sentence(clean, lang_name, allow_fallback_marker=False)
    if whole_trans is not None:
        return whole_trans

    # 2. Otherwise split by sentences and translate each part
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+|\n+", clean) if s.strip()]
    translated_parts = []
    for s in sentences:
        trans = _translate_single_sentence(s, lang_name, allow_fallback_marker=True)
        if trans:
            translated_parts.append(trans)

    return " ".join(translated_parts) if translated_parts else clean


def offline_summary(
    clauses: list[Any],
    language: str = "English",
    facts_and_findings: str = "",
) -> PlainSummary:
    """
    Create a local template summary without calling an AI provider.
    Accepts any supported language and produces genuine target-script text.
    Verbatim contract quotes are never translated or modified.
    """
    from src.ca.generation.summary import _template_summary

    language_key = str(language).strip().casefold()
    if language_key not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Unsupported language. Choose one of: "
            + ", ".join(SUPPORTED_LANGUAGES.values())
        )

    language_name = SUPPORTED_LANGUAGES[language_key]
    return _template_summary(
        clauses=clauses,
        language=language_name,
        facts_and_findings=facts_and_findings,
    )
