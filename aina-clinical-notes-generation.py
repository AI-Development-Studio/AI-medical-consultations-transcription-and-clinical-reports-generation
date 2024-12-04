import json
from typing import Dict, Optional
import re
from dataclasses import dataclass
from transformers import pipeline

@dataclass
class VisitSummary:
    """Data structure for medical visit summary"""
    patient_info: Dict[str, str]
    chief_complaint: str
    examination: str
    assessment: str
    treatment: str
    education: str
    follow_up: str
    notes: str

class AINAMedicalSummaryGenerator:
    def __init__(self):
        """Initialize AINA medical text processing components"""
        # Initialize text generation pipeline with AINA's Catalan model
        self.text_processor = pipeline(
            "text-generation",
            model="projecte-aina/FLOR-1.3B",
            truncation=True
        )
        
        # Medical section headers in Catalan
        self.section_headers = {
            "patient_info": "Informació del Pacient",
            "chief_complaint": "Motiu de Consulta i Història",
            "examination": "Exploració Física",
            "assessment": "Avaluació",
            "treatment": "Pla de Tractament",
            "education": "Educació del Pacient",
            "follow_up": "Pla de Seguiment",
            "notes": "Notes Addicionals"
        }
        
        # Medical terms dictionary
        self.medical_terms = {
            "fever": "febre",
            "pain": "dolor",
            "headache": "mal de cap",
            "blood pressure": "pressió arterial",
            "heart rate": "freqüència cardíaca",
            "medication": "medicació",
            "treatment": "tractament",
            "follow-up": "seguiment"
        }

    def _extract_section_content(self, text: str, section_patterns: list) -> str:
        """Extract content for a specific section using patterns"""
        content = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                content.append(match.group(1).strip())
        return "; ".join(content) if content else ""

    def _translate_medical_terms(self, text: str) -> str:
        """Translate common medical terms to Catalan"""
        translated = text.lower()
        for eng, cat in self.medical_terms.items():
            translated = re.sub(r'\b' + eng + r'\b', cat, translated)
        return translated

    def generate_summary(self, conversation: str, source_lang: str = "auto") -> str:
        """
        Generate structured medical visit summary in Catalan
        
        Args:
            conversation: Medical visit transcript
            source_lang: Source language code
        Returns:
            Formatted medical visit summary in Catalan markdown
        """
        # Extract information using section-specific patterns
        patterns = {
            "patient_info": [
                r"(?:patient|pacient)[:\s]+(.*?)(?=\n|$)",
                r"(?:age|edat)[:\s]+(.*?)(?=\n|$)",
            ],
            "chief_complaint": [
                r"(?:complaint|motiu|consulta)[:\s]+(.*?)(?=\n|$)",
                r"(?:symptoms|símptomes)[:\s]+(.*?)(?=\n|$)",
            ],
            # Add patterns for other sections
        }

        # Process each section
        summary = VisitSummary(
            patient_info={"info": self._extract_section_content(conversation, patterns["patient_info"])},
            chief_complaint=self._extract_section_content(conversation, patterns["chief_complaint"]),
            examination="",  # Add pattern extraction
            assessment="",   # Add pattern extraction
            treatment="",    # Add pattern extraction
            education="",    # Add pattern extraction
            follow_up="",    # Add pattern extraction
            notes=""        # Add pattern extraction
        )

        # Generate markdown format
        markdown_summary = f"""# Resum de la Visita

## {self.section_headers['patient_info']}
{summary.patient_info['info']}

## {self.section_headers['chief_complaint']}
{summary.chief_complaint}

## {self.section_headers['examination']}
{summary.examination}

## {self.section_headers['assessment']}
{summary.assessment}

## {self.section_headers['treatment']}
{summary.treatment}

## {self.section_headers['education']}
{summary.education}

## {self.section_headers['follow_up']}
{summary.follow_up}

## {self.section_headers['notes']}
{summary.notes}
"""

        return markdown_summary

def generate_visit_summary_using_aina(conversation: str, source_lang: str, target_lang: str = "ca") -> str:
    """
    Main function to generate medical visit summary using AINA
    
    Args:
        conversation: Medical visit transcript
        source_lang: Source language code
        target_lang: Target language code (default: Catalan)
    Returns:
        Formatted visit summary in Catalan
    """
    generator = AINAMedicalSummaryGenerator()
    return generator.generate_summary(conversation, source_lang)

# Example usage
if __name__ == "__main__":
    sample_conversation = """

[00:00:00 - 00:00:12] Other: hello today we are in manresa and we are with montse hello a follower of the project who proposed a collaboration today we will show you how to go to the doctor let's go let's go

[00:00:24 - 00:00:34] Doctor: good morning you are calling cap bonrepòs in a few moments one of our colleagues will assist you please stay on the line hello good morning

[00:00:34 - 00:00:40] Patient: hello good morning i'm not feeling well i was wondering if you could give me an appointment for today

[00:00:40 - 00:00:42] Doctor: yes what's wrong

[00:00:42 - 00:00:48] Patient: well for two days i've had mucus cough sore throat and fever

[00:00:49 - 00:00:50] Doctor: when did you check your fever

[00:00:51 - 00:00:54] Patient: last night and this morning

[00:00:55 - 00:00:56] Doctor: and what was it

[00:00:57 - 00:01:02] Patient: last night it was 37 and this morning 38

[00:01:03 - 00:01:05] Doctor: and since when have you been feeling ill

[00:01:05 - 00:01:05] Other: since

[00:01:05 - 00:01:06] Patient: for two days

[00:01:07 - 00:01:10] Doctor: okay well can you give me your id number please

[00:01:10 - 00:01:11] Patient: yes

[00:01:12 - 00:01:17] Patient: 74 524 (2) j for john

[00:01:18 - 00:01:24] Doctor: ah yes i now see that we don't have your date of birth can you tell me your date of birth please

[00:01:25 - 00:01:26] Patient: april 1st 1990

[00:01:28 - 00:01:34] Doctor: okay let's see would thursday at 3 work for you doctor lourdes casals will see you

[00:01:34 - 00:01:37] Patient: thursday but that's 3 days away

[00:01:38 - 00:01:42] Doctor: it's the first available slot for a visit with the primary care doctor we are saturated

[00:01:42 - 00:01:45] Patient: no no i can't wait that long i feel very sick

[00:01:45 - 00:01:50] Doctor: then it's better to go directly to the emergency room at the cap there you don't need an appointment

[00:01:50 - 00:01:52] Patient: but i need a sick note

[00:01:52 - 00:01:55] Doctor: when you go to the cap ask the doctor on duty to see you

[00:01:56 - 00:01:57] Patient: okay thanks

[00:01:57 - 00:01:58] Doctor: get well soon

[00:01:58 - 00:01:59] Patient: same to you goodbye

[00:02:01 - 00:02:39] Patient: hello good morning i wanted to be seen for emergency yes no problem do you have your health card yes yes yes very good what's wrong well i've been feeling sick for two days i have mucus and my throat hurts when swallowing last night i had a slight fever and this morning i took my temperature again and it was 38 and a half okay silvia then go to the waiting room and they'll call you soon okay

[00:02:40 - 00:02:44] Doctor: silvia claret yes come in please

[00:02:47 - 00:03:19] Patient: good morning good morning tell me what's wrong well i have a sore throat mucus fever and now my chest also hurts and i have trouble breathing when did you start feeling like this a couple of days ago you mentioned you have fever what temperature did you measure last night i checked it and it was 37 and this morning i took my temperature again and it was 38 and then anything else yes last night a paracetamol for the fever and another one this morning

[00:03:19 - 00:05:11] Doctor: okay then now we'll check your vitals very good first let's see if you have fever maybe lift your right arm like this i'll put the thermometer and now we'll check your blood pressure stretch out your left arm like this very good one moment like this perfect you'll feel some pressure very good the pressure is fine let's see the thermometer yes we can see that yes you still have fever but it's gone down you have 38 well and now i'll listen to you breathe deeply one moment please very good take a breath exhale another time breathe deeply and now breathe normally your lungs are congested now we'll check your throat open your mouth please say ah yes it's quite red do you have any contraindications or allergies no perfect then we need to do some tests a chest x-ray and a blood test to rule out anything serious now you'll have to wait in the room and my colleagues will call you for the tests see you soon okay see you soon

[00:05:11 - 00:05:11] Patient: see you soon

[00:05:21 - 00:05:52] Patient: while i wait for the test results i hope you're as well as i am how is your catalan learning going if you want to come practice your catalan with natives and other people like you catalan learners you can join our discord you can become members at isicatran.orgmembership and you'll have this advantage and many others we're waiting for you and now i think they'll call me again soon

[00:05:53 - 00:05:58] Doctor: hello again after seeing the tests i have to tell you that you have pneumonia

[00:05:59 - 00:06:00] Patient: oh wow what are you saying

[00:06:00 - 00:07:01] Doctor: regarding the blood test we'll have the results in a few days if we see anything that we need to discuss we'll call you by phone to give you the results you can also find them in the my health app you'll need to take antibiotics every 8 hours are you sure you don't have any contraindications or allergies no okay then you'll need to take one pill of this medication after breakfast lunch and dinner right now when you get home you can take the first one after lunch and in the evening again after dinner okay you'll also need to take paracetamol to control the fever as you've been doing until now understood especially you need to drink lots of fluids get rest and monitor your fever if after 5 days you don't improve go to the hospital emergency room understood are you going to work today

[00:07:02 - 00:07:04] Patient: no no i don't feel up to it

[00:07:04 - 00:07:13] Doctor: okay then i'll give you sick leave for 5 days which you can also find in the my health app now take good care

[00:07:13 - 00:07:14] Patient: okay thanks

[00:07:14 - 00:07:15] Doctor: get well soon

[00:07:15 - 00:07:16] Patient: and goodbye goodbye

[00:07:18 - 00:07:19] Doctor: and that's today's video

[00:07:19 - 00:07:35] Other: how are you silvia how are you feeling much better thank you we want to take this opportunity to thank eva for letting us use her reflexology and natural therapies center in manresa write in the comments how you are feeling and if you're feeling unwell what has
    """
    
    try:
        summary = generate_visit_summary_using_aina(sample_conversation, "en", "ca")
        print(summary)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
