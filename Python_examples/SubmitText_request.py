import requests
import os
import sys
from unidecode import unidecode
import unicodedata
import re

def submit_text_request(input_folder, bioconcept, output_file_session_number):
    pattern = r'[^0-9a-zA-Z\!\@\#\$\%\^\&\*\(\)\_\+\{\}\|\:\"\<\>\?\-\=\[\]\\;\'\,\.\/ \t\n\r]'
    unicode_to_regular = {}
    with open('lib/unicode.txt', 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            parts = line.split("\t")
            if len(parts) == 2:
                uni, reg = parts
                if reg == '-whitespace-':
                    unicode_to_regular[uni] = ' '
                else:
                    unicode_to_regular[uni] = reg
    
    with open(output_file_session_number, 'w', encoding='utf-8') as output_file:
        for filename in os.listdir(input_folder):
            if filename.startswith('.'):
                continue  # Skip hidden files
            text_str = ''
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file_input:
                for line in file_input:
                    normalized_line = unicodedata.normalize('NFC', line)
                    for uni, reg in unicode_to_regular.items():
                        normalized_line = normalized_line.replace(uni, reg)
                    normalized_line = unidecode(normalized_line)  # Assuming unidecode does similar work to Perl's version
                    text_str += normalized_line
            text_str = re.sub(pattern, ' ', text_str)
            url = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/request.cgi"
            response = requests.post(url, data={'text': text_str, 'bioconcept': bioconcept})
            
            if response.status_code == 200:
                session_number = response.json().get('id', '')
                print(f"Thanks for your submission. The session number is: {session_number}")
                output_file.write(f"{session_number}\t{filename}\n")
            else:
                print(f"Error: HTTP {response.status_code} for {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python submit_text_request.py [Inputfolder] [Bioconcept] [outputfile_SessionNumber]")
        print("\t[Inputfolder]: a folder with files to submit")
        print("\t[Bioconcept]: Gene, Disease, Chemical, Species, Mutation, and All.")
        print("\t[outputfile_SessionNumber]: output file to save session numbers.")
        print("Example: python submit_text_request.py input All SessionNumber.txt")
    else:
        input_folder = sys.argv[1]
        bioconcept = sys.argv[2]
        output_file_session_number = sys.argv[3]
        submit_text_request(input_folder, bioconcept, output_file_session_number)
